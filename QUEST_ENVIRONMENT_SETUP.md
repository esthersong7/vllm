# Quest Environment Setup Guide

다른 서버에서 현재 quest 환경을 재현하기 위한 가이드입니다.

## 환경 정보

### 기본 정보
- **Python 버전**: 3.10.19
- **PyTorch 버전**: 2.9.1+cu128
- **CUDA 버전**: 12.8 (PyTorch), 12.9 (시스템 nvcc)
- **Conda 환경 이름**: quest

### 주요 패키지
- vllm: 0.15.0rc2.dev58+g07ea184f0 (editable mode - /home2/esthersong7/vllm)
- torch: 2.9.1+cu128
- cuda-python: 13.1.1
- cupy-cuda12x: 13.6.0
- nvidia-cudnn-cu12: 9.10.2.21

## 재현 방법

### 방법 1: Conda Environment 전체 재현 (권장)

가장 정확한 재현을 위해 conda environment.yml 사용:

```bash
# 1. environment.yml로 환경 생성
conda env create -f quest_environment.yml

# 2. 환경 활성화
conda activate quest

# 3. vllm을 editable mode로 재설치
cd /path/to/vllm  # vllm git repository 경로
pip install -e .
```

### 방법 2: 처음부터 단계별 설치

```bash
# 1. Conda 환경 생성 (Python 3.10)
conda create -n quest python=3.10.19 -y
conda activate quest

# 2. PyTorch 설치 (CUDA 12.8)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. vllm git clone 및 editable install
git clone <your-vllm-repo-url>
cd vllm
pip install -e .

# 4. Quest 추가 의존성 설치
pip install -r /path/to/quest_requirements.txt

# 주의: vllm이 이미 설치되어 충돌할 수 있으므로, 
# requirements.txt에서 vllm 라인은 제외하거나 건너뛰기
```

### 방법 3: Requirements.txt만 사용 (빠른 재현)

```bash
# 1. Conda 환경 생성
conda create -n quest python=3.10.19 -y
conda activate quest

# 2. Requirements.txt로 패키지 일괄 설치
pip install -r quest_requirements.txt

# 3. vllm은 editable mode로 재설치
cd /path/to/vllm
pip install -e .
```

## 파일 설명

- **quest_environment.yml**: Conda 환경 전체 스냅샷 (conda + pip 패키지 포함)
- **quest_requirements.txt**: pip freeze 결과 (모든 pip 패키지와 정확한 버전)

## 주의사항

### vllm Editable Install
- vllm은 editable mode (`pip install -e .`)로 설치되어 있습니다
- Git repository에서 clone 후 직접 설치해야 합니다
- requirements.txt의 vllm 항목은 로컬 경로를 참조하므로 무시하고, 직접 설치하세요

### CUDA 버전
- PyTorch는 CUDA 12.8용으로 빌드되었습니다
- 시스템 nvcc는 12.9이지만, PyTorch CUDA 12.8과 호환됩니다
- 다른 서버에서는 CUDA 12.x 이상이 설치되어 있어야 합니다

### 다른 서버의 CUDA 버전이 다른 경우

#### CUDA 버전 확인
먼저 대상 서버의 CUDA 버전을 확인:
```bash
nvcc --version  # 시스템 CUDA 버전
nvidia-smi      # Driver가 지원하는 최대 CUDA 버전
```

#### 케이스별 대응 방법

**케이스 1: 시스템 CUDA가 12.x 이상인 경우** (예: CUDA 12.9, 13.x)
- ✅ **문제 없음**: PyTorch CUDA 12.8은 하위 호환성이 있어 작동합니다
- 그대로 `quest_environment.yml` 또는 `quest_requirements.txt` 사용

**케이스 2: 시스템 CUDA가 11.x인 경우** (예: CUDA 11.8)
- ⚠️ **PyTorch 재설치 필요**: CUDA 버전에 맞는 PyTorch 설치
```bash
# 환경 생성 후
conda create -n quest python=3.10.19 -y
conda activate quest

# CUDA 11.8용 PyTorch 설치
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치 (torch 제외)
grep -v "^torch==" quest_requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt

# vllm editable install
cd /path/to/vllm
pip install -e .
```

**케이스 3: CUDA 10.x 이하인 경우**
- ❌ **지원 불가**: PyTorch 2.9.1은 CUDA 11.8 이상 필요
- 서버의 CUDA 업그레이드 필요 또는 PyTorch 버전 다운그레이드 고려

**케이스 4: CUDA가 설치되지 않은 경우 (CPU only)**
- CPU 버전으로 테스트는 가능하나, vllm은 GPU 필수
```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

#### PyTorch CUDA 버전별 설치 명령어

| 시스템 CUDA | PyTorch 설치 명령어 |
|------------|-------------------|
| CUDA 12.x  | `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128` |
| CUDA 11.8  | `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu118` |
| CPU only   | `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu` |

#### 호환성 검증

설치 후 CUDA가 제대로 인식되는지 확인:
```bash
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
```

출력 예시:
```
PyTorch CUDA: 12.8
Available: True
Device Count: 8
```

### Git Repository 의존성
- vllm이 editable mode로 설치되어 있으므로, 코드 변경사항이 즉시 반영됩니다
- Git에서 올바른 브랜치/커밋을 체크아웃했는지 확인하세요

## 검증

환경이 올바르게 설정되었는지 확인:

```bash
# Python 버전 확인
python --version  # Python 3.10.19 예상

# PyTorch 및 CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# vllm 설치 확인
python -c "import vllm; print(f'vllm version: {vllm.__version__}')"

# vllm이 editable mode인지 확인
pip show vllm | grep Location
```

## 트러블슈팅

### CUDA 관련 에러

**"CUDA driver version is insufficient"**
- nvidia-smi로 Driver 버전 확인
- Driver가 지원하는 CUDA 버전보다 높은 PyTorch CUDA 버전을 사용 중
- 해결: Driver 업그레이드 또는 낮은 CUDA 버전의 PyTorch 설치

**"torch.cuda.is_available() returns False"**
```bash
# 원인 진단
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
nvcc --version
nvidia-smi

# CUDA 버전 불일치 시 PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu118  # 적절한 CUDA 버전으로
```

**vllm 빌드 중 CUDA 에러**
- vllm은 컴파일 시 시스템의 nvcc를 사용
- nvcc가 PATH에 있는지 확인: `which nvcc`
- 없으면 CUDA toolkit 설치 또는 PATH 추가:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA Out of Memory
- GPU 메모리가 부족한 경우, 모델 크기나 배치 사이즈 조정 필요
- `nvidia-smi`로 GPU 메모리 사용량 확인

### 버전 충돌
- requirements.txt 설치 시 버전 충돌이 발생하면, environment.yml 사용 권장
- 특정 패키지만 문제가 되면 해당 패키지를 제외하고 설치 후 수동 설치
- PyTorch 관련 패키지는 항상 먼저 설치 후 나머지 설치

### vllm import 에러
- vllm을 editable mode로 재설치: `cd vllm && pip install -e .`
- C++ extension 빌드가 필요한 경우 시간이 걸릴 수 있음 (수 분 ~ 십수 분)
- 빌드 로그를 확인하여 CUDA 관련 에러가 있는지 체크

## 생성 날짜
2026-02-12
