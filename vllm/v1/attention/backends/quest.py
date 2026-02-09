# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with Quest (decode-only optimization for long context).

Quest is designed for sequential inference (prefill-then-decode), not chunked prefill.
Requires enable_chunked_prefill=False in vLLM config.

Integration approach:
- vLLM provides multi-GPU framework (TP/PP), scheduling, input/output handling
- Quest manages its own KV cache + metadata cache (paged, self-allocated)
- Quest provides sparse attention computation
- Sequential flow: complete prefill → decode (no chunked prefill)

TP support:
- Quest automatically works with vLLM's TP (num_kv_heads already split per GPU)
- No Quest code changes needed for multi-GPU
- NVLink optimizations handled by vLLM's NCCL
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)

# Check Quest availability at import time
try:
    from vllm.v1.attention.ops.quest_controller import VllmQuestController
    QUEST_AVAILABLE = True
except ImportError as e:
    QUEST_AVAILABLE = False
    logger.warning(f"Quest backend unavailable: {e}")
    logger.warning("Install Quest: pip install -e /path/to/quest")


@dataclass
class QuestConfig:
    """Configuration for Quest attention backend.
    
    Quest is optimized for long-context decode with sparse attention.
    Token budget controls sparsity: higher budget = more accurate but slower.
    """
    page_size: int = 16
    """KV cache page size (tokens per page). Default: 16"""
    
    token_budget: int = 512
    """Token budget for sparse attention. Controls sparsity ratio.
    For 32k context: budget=512 means ~64x compression (32k/512).
    Higher budget = more tokens selected = more accurate.
    """
    
    max_seq_len: int = 131072  # 128k
    """Maximum sequence length for KV cache allocation. Default: 128k"""
    
    skip_layers: int = 2
    """Number of initial layers to skip sparse attention (use full). Default: 2"""
    
    @classmethod
    def from_env_or_default(cls) -> "QuestConfig":
        """Create config from environment variables or use defaults."""
        import os
        return cls(
            page_size=int(os.getenv("VLLM_QUEST_PAGE_SIZE", "16")),
            token_budget=int(os.getenv("VLLM_QUEST_TOKEN_BUDGET", "512")),
            max_seq_len=int(os.getenv("VLLM_QUEST_MAX_SEQ_LEN", "131072")),
            skip_layers=int(os.getenv("VLLM_QUEST_SKIP_LAYERS", "2")),
        )
    
    @property
    def page_budget(self) -> int:
        """Page budget = token_budget / page_size"""
        return self.token_budget // self.page_size


@dataclass
class QuestMetadata:
    """Simple metadata for Quest attention.
    
    Quest uses sequential inference: complete prefill, then decode.
    No mixed batching (no chunked prefill).
    """
    is_prefill: bool
    """True if this is prefill phase, False if decode phase"""
    
    seq_len: int
    """Current sequence length"""
    
    slot_mapping: torch.Tensor
    """Tensor for writing K/V to cache. Shape: [num_tokens]"""
    
    num_prefill_tokens: int
    """Number of prefill tokens"""
    
    num_prefilled_tokens: int
    """Number of tokens already prefilled (for continuation)"""
    
    # vLLM block table info (for Quest kernel indices)
    num_blocks: int
    """Total number of KV cache blocks allocated"""
    
    layer_idx: int = 0
    """Layer index (0 to num_layers-1). Set by build_attn_metadata per layer."""


class QuestMetadataBuilder(AttentionMetadataBuilder["QuestMetadata"]):
    """Metadata builder for Quest attention.
    
    Quest requires sequential inference (no chunked prefill).
    """
    
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        
        # Quest requires sequential inference
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "Quest backend requires enable_chunked_prefill=False. "
                "Quest is designed for sequential inference: prefill entire sequence, then decode. "
                "Set chunked_prefill_enabled=False in your SchedulerConfig."
            )
        
        self.page_size = kv_cache_spec.block_size
        logger.info(f"Quest backend initialized with page_size={self.page_size}")
        
    @classmethod
    def get_cudagraph_support(
        cls: type["QuestMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Quest does not support CUDA graphs (begin_forward/end_forward pattern)."""
        return AttentionCGSupport.NEVER
    
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QuestMetadata:
        """Build Quest metadata.
        
        Quest uses sequential inference:
        - Prefill: max_query_len > 1 (entire prompt at once)
        - Decode: max_query_len == 1 (one token at a time)
        
        Extracts vLLM block information for Quest kernels.
        Note: layer_idx will be set by build_attn_metadata per-layer.
        """
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        
        is_prefill = max_query_len > 1
        
        # For Quest's sequential inference:
        # - Prefill: num_prefill_tokens = num_actual_tokens, num_prefilled_tokens = 0
        # - Decode: num_prefill_tokens = 0, num_prefilled_tokens = max_seq_len - 1
        if is_prefill:
            num_prefill_tokens = num_actual_tokens
            num_prefilled_tokens = 0
        else:
            num_prefill_tokens = 0
            num_prefilled_tokens = max_seq_len - num_actual_tokens
        
        # Calculate number of blocks from slot_mapping
        # vLLM allocates blocks, we just need to know how many
        slot_mapping = common_attn_metadata.slot_mapping
        num_blocks = (slot_mapping.max().item() // self.page_size + 1) if slot_mapping.numel() > 0 else 0
        
        return QuestMetadata(
            is_prefill=is_prefill,
            seq_len=max_seq_len,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_prefilled_tokens=num_prefilled_tokens,
            num_blocks=num_blocks,
            layer_idx=0,  # Default 0, will be replaced by build_attn_metadata
        )


class QuestBackend(AttentionBackend):
    """Quest attention backend for long-context decode optimization.
    
    Optimized for DGX H100 multi-GPU inference with long contexts (32k-1M tokens).
    Uses sparse attention with configurable token budget.
    
    Quest manages its own KV cache (not vLLM's KV cache system).
    Multi-GPU (TP) works automatically via vLLM's infrastructure.
    """
    
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
    
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """Supported page sizes for Quest."""
        return [16, 32]  # Quest typically uses page_size=16
    
    @staticmethod
    def get_name() -> str:
        return "QUEST"
    
    @staticmethod
    def get_impl_cls() -> type["QuestImpl"]:
        return QuestImpl
    
    @staticmethod
    def get_builder_cls() -> type["QuestMetadataBuilder"]:
        return QuestMetadataBuilder
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """KV cache shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        
        Quest manages its own KV cache separately, but vLLM still allocates
        this cache. Use low gpu_memory_utilization (e.g., 0.3) to leave room
        for Quest's standalone KV cache allocation.
        """
        return (num_blocks, 2, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Stride order for KV cache."""
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        else:
            return (0, 1, 2, 3, 4)
    
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """Supported head sizes."""
        return [64, 128, 256]
    
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """Quest requires Ampere or newer (SM 8.0+)."""
        return capability >= DeviceCapability(8, 0)


class QuestImpl(AttentionImpl):
    """Quest attention implementation with standalone KV cache management.
    
    Manages Quest's own KV cache, metadata cache, and sparse attention logic.
    Follows Quest's original approach: self-managed paged KV cache.
    
    Multi-GPU (TP) support:
    - vLLM splits num_kv_heads across GPUs automatically
    - Quest receives already-split heads (e.g., 8 heads on each of 4 GPUs)
    - No Quest code changes needed for TP
    - Works seamlessly with NVLink on DGX H100
    """
    
    can_return_lse_for_decode: bool = False
    
    # Class-level shared controller (all layers share one controller)
    _shared_controller: Optional["VllmQuestController"] = None
    _controller_initialized: bool = False
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if not QUEST_AVAILABLE:
            raise ImportError(
                "Quest backend is not available. Install Quest:\n"
                "  pip install -e /path/to/quest"
            )
        
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        if alibi_slopes is not None:
            raise NotImplementedError("Quest does not support ALiBi")
        if sliding_window is not None:
            raise NotImplementedError("Quest does not support sliding window")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Quest only supports decoder-only attention")
        if sinks is not None:
            raise NotImplementedError("Quest does not support attention sinks")
        if logits_soft_cap is not None:
            raise NotImplementedError("Quest does not support logits soft cap")
        
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        
        # Quest configuration
        self.quest_config = QuestConfig.from_env_or_default()
        
        # Use shared controller (same for all layers)
        # Individual layers don't create their own controllers
        
        logger.info(
            f"Quest attention initialized: "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_size={head_size}, "
            f"page_size={self.quest_config.page_size}, token_budget={self.quest_config.token_budget}"
        )
    
    def _init_controller(
        self,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize shared Quest controller (one controller for all layers)."""
        # Only initialize once (shared across all layers)
        if QuestImpl._controller_initialized:
            return
        
        page_budget = self.quest_config.token_budget // self.quest_config.page_size
        
        QuestImpl._shared_controller = VllmQuestController(
            num_layers=num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_size,
            page_size=self.quest_config.page_size,
            page_budget=page_budget,
            max_seq_len=self.quest_config.max_seq_len,
            dtype=dtype,
            device=device,
            num_kv_heads=self.num_kv_heads,
            share_pages=True,  # GQA optimization
        )
        
        QuestImpl._controller_initialized = True
        
        logger.info(
            f"Initialized Quest controller: "
            f"page_budget={page_budget}, max_seq_len={self.quest_config.max_seq_len}, "
            f"skip_layers={self.quest_config.skip_layers}, "
            f"num_kv_heads={self.num_kv_heads} (TP-split)"
        )
    

    
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: QuestMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Quest attention (Quest manages KV cache).
        
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: vLLM paged cache (NOT USED - Quest manages its own)
            attn_metadata: Quest metadata with layer_idx
            
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        
        if attn_metadata is None:
            # Profiling run
            # logger.info("QuestImpl forward() called without metadata, returning zeros.")
            return output.fill_(0)
        
        # Initialize controller on first call (shared across all layers)
        if not QuestImpl._controller_initialized:
            import os
            num_layers = int(os.environ.get("VLLM_QUEST_NUM_LAYERS", "32"))
            logger.info(f"QuestImpl initializing shared controller ")
            self._init_controller(
                num_layers=num_layers,
                device=query.device,
                dtype=query.dtype,
            )
        
        # Get layer index from metadata (set by build_attn_metadata)
        layer_idx = attn_metadata.layer_idx
        seq_len = attn_metadata.seq_len
        is_prefill = attn_metadata.is_prefill


        num_tokens = key.shape[0]  # vLLM padded size
        current_kv_len = QuestImpl._shared_controller.kv_cache.seqlen
        logger.info(f"[Quest L{layer_idx}] forward(): is_prefill={is_prefill}, seq_len={seq_len}, num_tokens={num_tokens}, current_kv_len={current_kv_len}")
        
        
        # Layer 0 ONLY: prepare metadata (allocate KV cache pages) + begin_forward
        if layer_idx == 0:
            num_tokens = key.shape[0]  # vLLM padded size
            current_kv_len = QuestImpl._shared_controller.kv_cache.seqlen
            
            # logger.info(f"[Quest L{layer_idx}] {'PREFILL' if is_prefill else 'DECODE'}: "
            #            f"num_tokens={num_tokens}, seq_len={seq_len}, current_kv_len={current_kv_len}")
            
            # Detect new sequence: empty cache or seq reset
            if is_prefill:
                if current_kv_len == 0 or current_kv_len >= seq_len:
                    logger.info(f"[Quest] New sequence, clean states: (kv_len={current_kv_len} → 0)")
                    QuestImpl._shared_controller.clean_states()                         #################이걸왜하노?
                    current_kv_len = 0  
                
                # Allocate only actual NEW tokens (not padded size)
                actual_new_tokens = seq_len - current_kv_len
            else:
                # Decode: always 1 new token
                actual_new_tokens = 1
            
            # Quest model-level operations (once per step)
            QuestImpl._shared_controller.prepare_metadata(actual_new_tokens)
            logger.info(f"[Quest] After prepare_metadata w/ actual new tokens = {actual_new_tokens}: kv_seqlen={QuestImpl._shared_controller.kv_cache.seqlen}")
            
            # Skip layers logic: begin_forward with high budget for first N layers
            skip_layers = self.quest_config.skip_layers
            if skip_layers > 0:
                # First skip_layers use full attention (high page budget)
                QuestImpl._shared_controller.set_page_budget(1024 * 1024)
                QuestImpl._shared_controller.begin_forward(actual_new_tokens)
            else:
                # Normal: all layers use sparse attention
                QuestImpl._shared_controller.begin_forward(actual_new_tokens)
            
            logger.info(f"[Quest L{layer_idx}] After prepare: kv_seqlen={QuestImpl._shared_controller.kv_cache.seqlen}")
        
        # Skip layers transition: switch from full to sparse attention
        skip_layers = self.quest_config.skip_layers
        if skip_layers > 0 and layer_idx == skip_layers:
            logger.info(f"[Quest L{layer_idx}] Switching to sparse attention (skip_layers={skip_layers})")
            QuestImpl._shared_controller.end_forward()
            page_budget = self.quest_config.token_budget // self.quest_config.page_size
            QuestImpl._shared_controller.set_page_budget(page_budget)
            # begin_forward again with normal budget, but update_tensor=False (metadata already set)
            if is_prefill:
                actual_new_tokens = seq_len - QuestImpl._shared_controller.kv_cache.seqlen + key.shape[0]
            else:
                actual_new_tokens = 1
            QuestImpl._shared_controller.begin_forward(actual_new_tokens, update_tensor=False)
        
        # ✅ Append KV: slice to actual length (remove vLLM padding)
        if is_prefill:
            # Prefill: use only actual tokens (seq_len), not padded (num_tokens)
            actual_len = min(seq_len, key.shape[0])
            key_actual = key[:actual_len].contiguous()
            value_actual = value[:actual_len].contiguous()
        else:
            # Decode: always 1 token (no padding)
            key_actual = key.contiguous()
            value_actual = value.contiguous()
        
        QuestImpl._shared_controller.append_kv_and_metadata(
            key_actual, value_actual, layer_idx, is_prefill
        )
        
        try:
            # Attention computation
            if is_prefill:
                # ✅ Prefill: use actual seq_len for query (remove padding)
                query_actual = query[:min(seq_len, query.shape[0])].contiguous()
                attn_output = self._quest_prefill(query_actual, layer_idx)
            else:
                # Decode: check if sparse is needed
                need_sparse = QuestImpl._shared_controller.need_estimate()
                if layer_idx == 0:
                    logger.info(f"[Quest L0 Decode] need_sparse={need_sparse}, "
                               f"num_pages={len(QuestImpl._shared_controller.kv_cache.indicies)}, "
                               f"page_budget={QuestImpl._shared_controller.inference_page_budget}")
                
                if need_sparse:
                    # Sparse decode
                    attn_output = QuestImpl._shared_controller.sparse_decode_attention(
                        query, layer_idx, use_full_attention=False
                    )
                else:
                    # Full decode
                    attn_output = QuestImpl._shared_controller.sparse_decode_attention(
                        query, layer_idx, use_full_attention=True
                    )
            
            # ✅ Copy only actual output
            if is_prefill:
                output[:attn_output.shape[0]].copy_(attn_output)
            else:
                output.copy_(attn_output)
            
        finally:
            # Last layer: end_forward
            import os
            num_layers = int(os.environ.get("VLLM_QUEST_NUM_LAYERS", "32"))
            if layer_idx == num_layers - 1:
                QuestImpl._shared_controller.end_forward()
        
        return output
    

    def _quest_prefill(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Quest prefill using Quest's own paged KV cache.
        
        Args:
            query: [num_tokens, num_heads, head_dim]
            layer_idx: Layer index
            
        Returns:
            [num_tokens, num_heads, head_dim]
        """
        try:
            import quest._kernels as _kernels
            from quest.utils.utils import TensorLayout
            
            query = query.contiguous()
            
            # Use Quest's KV cache (shared controller)
            kv_indices = QuestImpl._shared_controller.kv_indices_with_last
            last_page_len = QuestImpl._shared_controller.kv_last_page_len
            paged_kv_data = QuestImpl._shared_controller.kv_cache.buf_layer(layer_idx)
            
            # Call Quest prefill kernel
            attn_output = _kernels.prefill_with_paged_kv_cache(
                query.contiguous(),
                paged_kv_data,
                kv_indices,
                last_page_len,
                True,  # causal
                TensorLayout.NHD,
                False,  # allow_fp16_qk_reduction
                1.0,   # rope_scale
                1e4,   # rope_theta
            )
            
            return attn_output
        except Exception as e:
            logger.warning_once(f"Quest prefill failed ({e}), using zeros fallback.")
            return torch.zeros_like(query)
