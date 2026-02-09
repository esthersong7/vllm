# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-specific Quest controller for sparse attention.

This controller adapts Quest's sparse attention for vLLM's KV cache management.
Unlike Quest's original InferenceController which manages its own KV cache,
this controller only manages metadata cache and coordinates with vLLM's KV cache.
"""

import torch
from typing import Optional

try:
    import quest._kernels as _kernels
    from quest.utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper
    from quest.utils.kv_cache import KvCache
    from quest.utils.utils import TensorLayout
    QUEST_AVAILABLE = True
except ImportError as e:
    QUEST_AVAILABLE = False
    _IMPORT_ERROR = e


class VllmQuestController:
    """Quest controller for vLLM integration (standalone KV cache).
    
    This controller manages both KV cache and metadata cache independently
    following Quest's original approach.
    
    Key differences from vLLM's standard KV cache:
    - Quest manages its own paged KV cache + metadata cache
    - Uses Quest's sparse attention kernels
    - Coordinates with vLLM for scheduling and multi-GPU
    
    vLLM provides:
    - Multi-GPU framework (TP/PP) 
    - Scheduling
    - Input/output handling
    
    Quest manages:
    - KV cache (paged, self-allocated)
    - Metadata cache (min/max per page)
    - Sparse attention (estimate→topk→decode)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        page_budget: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        num_kv_heads: Optional[int] = None,
        share_pages: bool = True,
    ):
        """Initialize vLLM Quest controller.
        
        Args:
            num_layers: Number of model layers
            num_heads: Number of query heads
            head_dim: Head dimension
            page_size: KV cache page size (must match vLLM block size)
            page_budget: Token budget for sparse attention
            max_seq_len: Maximum sequence length
            dtype: Data type
            device: Device
            num_kv_heads: Number of KV heads (for GQA), defaults to num_heads
            share_pages: Whether to share page selection across query heads in same KV group
        """
        if not QUEST_AVAILABLE:
            raise ImportError(
                f"Quest library is not available. Please install Quest:\n"
                f"  cd /path/to/quest && pip install -e .\n"
                f"Original error: {_IMPORT_ERROR}"
            )
        
        # GQA support
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.group_size = num_heads // self.num_kv_heads
        self.share_pages = share_pages
        
        # KV cache (Quest manages this)
        # vLLM handles batches, so allocate 10x capacity for concurrent sequences
        total_cache_capacity = max_seq_len * 10
        self.kv_cache = KvCache(
            num_layers=num_layers,
            num_heads=self.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=total_cache_capacity,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        
        # Metadata cache (Quest manages this)
        # Metadata is per-page, also need 10x capacity for batching
        max_pages = (total_cache_capacity + page_size - 1) // page_size
        self.metadata_cache = KvCache(
            num_layers=num_layers,
            num_heads=self.num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_pages,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        
        self.layout = TensorLayout.NHD
        self.device = device
        self.dtype = dtype
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._page_budget = page_budget
        self._max_page_limit = 1024 * 1024
        
        # Decode handler
        self._decode_handler = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")
        
        # State variables (set by vLLM during forward)
        self.kv_indices_with_last: Optional[torch.Tensor] = None
        self.kv_indices_without_last: Optional[torch.Tensor] = None
        self.metadata_indices: Optional[torch.Tensor] = None
        self.kv_last_page_idx: Optional[int] = None
        self.metadata_last_page_idx: Optional[int] = None
        self.kv_last_page_len: int = page_size
        
        self.kv_indptr_for_append: Optional[torch.Tensor] = None
        self.metadata_indptr_for_append: Optional[torch.Tensor] = None
        self.kv_indptr_for_approx_decode: Optional[torch.Tensor] = None
        
        self.inference_page_budget: Optional[int] = None
        
        # TopK buffers
        self.topk_dout_buffer: Optional[torch.Tensor] = None
        self.topk_dindices_buffer: Optional[torch.Tensor] = None
        self.topk_buf: Optional[torch.Tensor] = None
        
        self._forward_begun = False
    
    def set_page_budget(self, page_budget: int):
        """Set page budget for sparse attention."""
        self._page_budget = page_budget
    
    def prepare_metadata(self, seq_len: int):
        """Allocate space for new tokens (Quest original approach).
        
        Called once per forward pass before processing all layers.
        
        Args:
            seq_len: Number of NEW tokens to add (not padded size)
        """
        # ✅ Only allocate for actual new tokens
        if seq_len <= 0:
            return
        
        # Allocate entry for tokens in KV cache
        appended_new_pages = self.kv_cache.append_seq(seq_len)
        # Allocate entry for metadata (one metadata entry per KV page)
        _ = self.metadata_cache.append_seq(appended_new_pages)
    
    def begin_forward(self, num_tokens: int, update_tensor: bool = True):
        """Prepare for forward pass (Quest original approach).
        
        Sets up indices and buffers for kernel calls.
        
        Args:
            num_tokens: Number of NEW tokens being added (prefill: N, decode: 1)
                       Used to detect prefill (>1) vs decode (=1) mode
            update_tensor: Whether to update tensor buffers (set False for layer sensitivity)
        """
        # Allocate tensor in advance
        if update_tensor:
            self.kv_indptr_for_append = torch.tensor(
                [0, len(self.kv_cache.indicies)], dtype=torch.int32, device=self.device
            )
            self.metadata_indptr_for_append = torch.tensor(
                [0, len(self.metadata_cache.indicies)], dtype=torch.int32, device=self.device
            )
            self.kv_last_page_idx = self.kv_cache.indicies[-1]
            self.metadata_last_page_idx = self.metadata_cache.indicies[-1]
            self.kv_last_page_len = self.kv_cache.last_page_len
        
        if num_tokens > 1:
            # Prefill: multiple tokens being added
            if update_tensor:
                self.kv_indices_with_last = torch.tensor(
                    self.kv_cache.indicies, dtype=torch.int32, device=self.device
                )
                self.metadata_indices = torch.tensor(
                    self.metadata_cache.indicies, dtype=torch.int32, device=self.device
                )
        else:
            # Decode: single token being added
            cur_page_nums = len(self.kv_cache.indicies)
            assert cur_page_nums > 1, f"Need at least 2 pages for decode, got {cur_page_nums}"
            
            if update_tensor:
                # KV indices with last page
                self.kv_indices_with_last = torch.tensor(
                    self.kv_cache.indicies, dtype=torch.int32, device=self.device
                )
                
                # KV indices without last (for topk - repeated for each KV head)
                self.kv_indices_without_last = torch.tensor(
                    self.kv_cache.indicies[:-1], dtype=torch.int32, device=self.device
                ).repeat(self.num_kv_heads, 1)
                
                # Metadata indices
                self.metadata_indices = torch.tensor(
                    self.metadata_cache.indicies, dtype=torch.int32, device=self.device
                )
            
            # Page budget for topk and decode
            self.inference_page_budget = min(self._page_budget, cur_page_nums)
            
            # Indptr for decode (exclude last page)
            self.kv_indptr_for_approx_decode = torch.tensor(
                [0, self.inference_page_budget - 1], dtype=torch.int32, device=self.device
            )
            
            # Allocate topk buffers
            buffer_heads = self.num_kv_heads if self.share_pages else self.num_heads
            self.topk_dout_buffer = torch.zeros(
                (buffer_heads, self.inference_page_budget - 1), dtype=self.dtype, device=self.device
            )
            self.topk_dindices_buffer = torch.zeros(
                (buffer_heads, self.inference_page_budget - 1), dtype=torch.int32, device=self.device
            )
            self.topk_buf = torch.zeros(
                (buffer_heads, 8192 * 2 * (2+4) // 2 // 48), dtype=self.dtype, device=self.device
            )
            
            # Initialize decode handler
            self._decode_handler.begin_forward(
                self.kv_indptr_for_approx_decode,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                self.dtype
            )
        
        self._forward_begun = True
    
    def end_forward(self):
        """Clean up after forward pass."""
        if self._forward_begun:
            self._decode_handler.end_forward()
            self._forward_begun = False
    
    def need_estimate(self) -> bool:
        """Check if sparse attention (estimate→topk→decode) is needed.
        
        Returns False if:
        - Page budget >= total pages (use full attention)
        - Not in decode mode
        
        Returns True if sparse attention should be used.
        """
        if self.inference_page_budget is None:
            return False
        
        if self.kv_indices_with_last is None:
            return False
        
        num_blocks = self.kv_indices_with_last.shape[0]
        return num_blocks > self.inference_page_budget
    
    def append_kv_and_metadata(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        is_prefill: bool,
    ):
        """Append both KV cache and metadata cache (Quest original approach).
        
        Args:
            key: Key tensor, shape [num_tokens, num_kv_heads, head_dim]
            value: Value tensor, shape [num_tokens, num_kv_heads, head_dim]
            layer_idx: Layer index
            is_prefill: True if prefill, False if decode
        """
        # Ensure tensors are contiguous for Quest kernels
        key = key.contiguous()
        value = value.contiguous()
        
        if is_prefill:
            _kernels.append_kv_cache_prefill(
                key,
                value,
                self.kv_cache.buf_layer(layer_idx),
                self.kv_indices_with_last,
                self.kv_indptr_for_append,
                self.kv_last_page_len,
                self.kv_last_page_idx,
                self.metadata_cache.buf_layer(layer_idx),
                self.metadata_indices,
                self.metadata_indptr_for_append,
                self.metadata_cache.last_page_len,
                self.metadata_last_page_idx,
                self.layout
            )
        else:
            _kernels.append_kv_cache_decode(
                key,
                value,
                self.kv_cache.buf_layer(layer_idx),
                self.kv_indices_with_last,
                self.kv_indptr_for_append,
                self.kv_last_page_len,
                self.kv_last_page_idx,
                self.metadata_cache.buf_layer(layer_idx),
                self.metadata_indices,
                self.metadata_indptr_for_append,
                self.metadata_cache.last_page_len,
                self.metadata_last_page_idx,
                self.layout
            )
    
    def estimate_attention_scores(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Estimate attention scores using metadata cache.
        
        Args:
            query: Query tensor, shape [1, num_heads, head_dim]
            layer_idx: Layer index
            
        Returns:
            Estimated scores, shape [num_heads, num_pages-1]
        """
        num_pages = self.metadata_cache.seqlen - 1
        
        # Ensure query is contiguous
        query = query.contiguous()
        
        # For GQA: compute per KV head, then expand to query heads
        q_squeezed = query.squeeze(0)  # [num_heads, head_dim]
        q_grouped = q_squeezed.view(self.num_kv_heads, self.group_size, -1)
        q_per_kv = q_grouped.mean(dim=1, keepdim=True).squeeze(1).unsqueeze(0).contiguous()  # [1, num_kv_heads, head_dim]
        
        # Estimate per KV head
        scores_kv = torch.empty((self.num_kv_heads, num_pages), dtype=query.dtype, device=query.device)
        _kernels.estimate_attn_score(
            q_per_kv,
            scores_kv,
            self.metadata_cache.buf_layer(layer_idx),
            self.metadata_indices,
            self.metadata_indptr_for_append,
            self.metadata_cache.last_page_len,
            self.metadata_last_page_idx,
            self.layout,
        )
        
        # Expand to all query heads
        scores = scores_kv.repeat_interleave(self.group_size, dim=0)
        return scores
    
    def topk_page_selection(self, estimated_scores: torch.Tensor):
        """Select top-k pages based on estimated scores.
        
        Results stored in self.topk_dindices_buffer.
        
        Args:
            estimated_scores: Scores from estimate_attention_scores()
        """
        page_budget = self.inference_page_budget - 1
        
        if self.share_pages and self.group_size > 1:
            # Aggregate scores across query heads in same KV group
            scores_grouped = estimated_scores.view(self.num_kv_heads, self.group_size, -1)
            scores_per_kv = scores_grouped.sum(dim=1)
            
            _kernels.topk_filtering(
                scores_per_kv,
                self.kv_indices_without_last,
                self.topk_dout_buffer,
                self.topk_dindices_buffer,
                self.topk_buf,
                page_budget,
            )
        else:
            _kernels.topk_filtering(
                estimated_scores,
                self.kv_indices_without_last,
                self.topk_dout_buffer,
                self.topk_dindices_buffer,
                self.topk_buf,
                page_budget,
            )
    
    def sparse_decode_attention(
        self,
        query: torch.Tensor,
        layer_idx: int,
        use_full_attention: bool = False,
    ) -> torch.Tensor:
        """Perform sparse decode attention (Quest original approach).
        
        Args:
            query: Query tensor, shape [1, num_heads, head_dim]
            layer_idx: Layer index
            use_full_attention: If True, use all pages; if False, do estimate→topk→sparse
            
        Returns:
            Output tensor, shape [1, num_heads, head_dim]
        """
        # Ensure query is contiguous
        query = query.contiguous()
        
        # If sparse, do estimate → topk first
        if not use_full_attention:
            # Step 1: Estimate scores using metadata cache
            estimated_scores = self.estimate_attention_scores(query, layer_idx)
            
            # Step 2: TopK page selection
            self.topk_page_selection(estimated_scores)
        
        # Get topk indices (or all indices for full attention)
        if use_full_attention:
            # Use all pages except last (last is handled separately by kernel)
            if self.share_pages and self.group_size > 1:
                topk_indices = self.kv_indices_without_last.repeat_interleave(self.group_size, dim=0)
            else:
                topk_indices = self.kv_indices_without_last
        else:
            # Use topk selection
            if self.share_pages and self.group_size > 1:
                topk_indices = self.topk_dindices_buffer.repeat_interleave(self.group_size, dim=0)
            else:
                topk_indices = self.topk_dindices_buffer
        
        # Output
        output = torch.empty_like(query).contiguous()
        
        # Call decode kernel (using Quest's KV cache)
        self._decode_handler.forward(
            query,
            output,
            self.kv_cache.buf_layer(layer_idx),  # Quest KV cache
            topk_indices,
            self.kv_indptr_for_approx_decode,
            self.kv_last_page_len,
            self.kv_last_page_idx,
            1.0,  # rope_scale
            1e4,  # rope_theta
        )
        
        return output
    
    def clean_states(self):
        """Clean up controller state."""
        self.kv_cache.release()
        self.metadata_cache.release()
