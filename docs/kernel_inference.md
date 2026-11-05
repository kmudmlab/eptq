# CUDA Inference Kernel

fp16으로 복원하지 않고 EPTQ 양자화 포맷에서 직접 추론하는 CUDA 커널 구현 문서입니다.

## 배경

기존 추론 흐름:
```
양자화된 파일 → dequantize_layer() → fp16 weight → torch.mm → 출력
```

커널 추론 흐름:
```
양자화된 파일 → EPTQLinear.forward() → 출력
               (CUDA 커널 내부에서 on-the-fly 복원 + matmul)
```

fp16 weight를 메모리에 올리지 않으므로 레이어당 **약 87%의 VRAM을 절약**합니다.  
LLaMA-7B의 경우 레이어 하나가 90 MB → 11 MB.

---

## 수학적 정의

양자화된 레이어의 forward pass:

```
y[b, 8m+k] = g[8m+k] · Σ_n { cb[Q[m,n]][k] · h[n] · x[b,n] }   (일반 위치)
           + g[8m+k] · Σ_i { O[i][k]         · h[n_i] · x[b,n_i] } (masked 위치)
           + bias[8m+k]
```

- `m` : Q의 행 인덱스 (8개 출력 차원 묶음)
- `k` : 묶음 내 오프셋 (0–7)
- `cb`: E8 또는 E8-D4 코드북
- `Q[m,n]`: `actual_index − 32768`으로 저장된 int16 인덱스 (sentinel 32767 = masked)
- `g`, `h`: row/column scale factor (fp16)
- `O`: critical weight 원본 값 (fp16, S로 정규화됨)

---

## 파일 구성

```
csrc/
  eptq_kernels.h    — C++ 함수 선언
  eptq_kernels.cu   — CUDA 커널 4개 + C++ 디스패치
  binding.cpp       — pybind11 바인딩

kernels/
  __init__.py       — JIT 로더 + Python API (e8_forward, e8d4_forward)
  linear.py         — EPTQLinear (nn.Module)
  patch.py          — load_quantized_model()

setup_cuda.py       — AOT 빌드 스크립트
```

---

## 커널 구조

### Kernel 1, 2: 메인 dequant + matvec

`e8_matvec_kernel` / `e8d4_matvec_kernel`

| 항목 | 값 |
|------|----|
| Grid | `(M, B)` — M = out_features/8, B = batch |
| Block | `(256)` = 8 warps |
| 역할 | Q의 한 행(m)을 담당하는 블록이 N개 입력 열을 256개 스레드로 분담 |

**실행 흐름 (스레드 하나 기준):**

1. `n = threadIdx.x, threadIdx.x+256, ...` 으로 열 순회
2. `Q[m,n]` 로드 → `uint16 idx = int16_value + 32768`
3. `idx == 65535`이면 skip (masked, correction kernel이 처리)
4. 코드북에서 8차원 벡터 로드
   - E8: `float4` (16 bytes) 로드 → 8개 fp16
   - E8-D4: `float2` × 2 (8 bytes × 2) → D4 벡터 두 개
5. `acc[0..7] += codeword[k] × h[n] × x[b,n]`
6. Warp shuffle reduction → shared memory inter-warp reduction
7. 첫 8개 스레드가 `y_f32[b, m*8+k] = acc[k] × g[m*8+k]` 기록

코드북 접근은 `__ldg()`(read-only cache)를 사용합니다. E8 코드북(~1 MB)은 RTX 4080의 L2(48 MB)에 상주합니다.

### Kernel 3: Masked weight 보정

`masked_correction_kernel`

- Grid: `(⌈n_masked/256⌉, B)`
- 스레드마다 masked 위치 하나를 처리
- `atomicAdd`로 float32 버퍼에 보정값 누적
- `percmask=0.002` 기본값에서 n_masked ≈ 전체 Q 원소의 0.2%

### Kernel 4: fp32 → fp16 변환

`f32_to_f16_kernel`

- float32 누산 버퍼를 fp16 출력으로 변환하고 bias를 더함
- float32 중간 버퍼를 쓰는 이유: `atomicAdd`가 fp16보다 fp32가 안정적이고 범용적임

---

## E8-D4 인덱스 인코딩

E8-D4는 int16 하나에 세 가지 정보를 비트 패킹합니다:

```
bits [15:14] = code_type  (0–3, D4 코드 종류)
bits [13: 7] = idx_i      (0–127, 첫 번째 4D 벡터)
bits [ 6: 0] = idx_j      (0–127, 두 번째 4D 벡터)
```

8차원 출력 = `concat(codes[code_type][idx_i], codes[code_type][idx_j])`

코드북 메모리 레이아웃(커널용): `(512, 4)` fp16 flat  
→ `codes[(code_type × 128 + idx) × 4 + k]`  
→ `float2` 단위(8 bytes)로 정렬 접근 가능

---

## Python API

### EPTQLinear

```python
from eptq.kernels.linear import EPTQLinear

# QuantizedLayerData에서 직접 생성
linear = EPTQLinear.from_quantized_data(
    quant_data,          # QuantizedLayerData
    codebook,            # E8: (≤65535, 8) fp16  /  E8-D4: (4, 128, 4) fp16
    quantizer_type,      # "e8_lattice" 또는 "e8_d4"
)

# nn.Linear와 동일한 인터페이스
y = linear(x)            # x: (..., in_features) fp16
```

**버퍼 목록** (`.to(device)` 시 자동 이동):

| 버퍼 | 타입 | 설명 |
|------|------|------|
| `Q` | int16 (M, N) | 양자화 인덱스 |
| `g` | fp16 (out,) | Row scale |
| `h` | fp16 (in,) | Column scale |
| `O_vals` | fp16 (n_masked, 8) | Critical weight 값 |
| `O_row` | int32 (n_masked,) | Masked 위치 행 인덱스 |
| `O_col` | int32 (n_masked,) | Masked 위치 열 인덱스 |
| `bias` | fp16 (out,) | 편향 (없으면 None) |

`_codebook`은 버퍼가 아닌 일반 속성으로 저장되어 레이어 간 **공유 참조**를 유지합니다. `_apply()` 오버라이드로 `model.to(device)` 시 함께 이동합니다.

### load_quantized_model

```python
from eptq.kernels.patch import load_quantized_model

model, tokenizer = load_quantized_model(
    quantized_dir="./quantized_opt-125m",
    device="cuda",
)
```

`reconstruct_model()`과 달리 weight를 fp16으로 복원하지 않습니다. 코드북은 모든 레이어가 공유하는 단 하나의 텐서로 로드됩니다.

---

## 빌드

**JIT 컴파일 (자동, 첫 실행 시):**

`kernels/__init__.py`가 `torch.utils.cpp_extension.load()`로 자동 컴파일합니다. 별도 설정 없이 동작합니다.

**AOT 빌드 (선택사항, 시작 속도 향상):**

```bash
python3 setup_cuda.py build_ext --inplace
```

컴파일된 `_eptq_cuda*.so`가 있으면 JIT 컴파일 없이 즉시 로드합니다.

**요구사항:** CUDA toolkit, PyTorch >= 2.0 (CUDA 빌드)

---

## 메모리 구성 비교

레이어 크기 `(out, in)` 기준:

| 항목 | fp16 (reconstruct) | 양자화 (kernel) |
|------|--------------------|----------------|
| Weight | `out × in × 2` bytes | `(out/8 × in) × 2` bytes |
| Scale | — | `(out + in) × 2` bytes |
| Critical | — | `n_masked × 8 × 2` bytes |
| **합계 (근사)** | `out × in × 2` | `out × in / 4 + …` |
| **압축률** | 1× | ~8× (약 87% 절약) |

LLaMA-7B FFN 레이어 (`11008 × 4096`) 예시:
- fp16: 90.2 MB
- 양자화: 11.5 MB

---

## 알려진 제한

- **Batch ≥ 8에서 느림**: cuBLAS 기반 fp16 matmul이 더 효율적. Prefill 단계에서는 `reconstruct_model()` 후 일반 추론 권장.
- **sm_70(Volta) 이상 필요**: warp shuffle 및 `__ldg()` 사용.
- **fp16 입력 필요**: `x`가 fp32이면 커널 내부에서 `.half()` 변환.
