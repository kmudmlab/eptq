# Inference 속도 벤치마크

CUDA inference kernel (`EPTQLinear`)과 fp16 복원 후 추론(`reconstruct_model`)의 속도를 비교합니다.

---

## End-to-End Token Generation 속도 (A100)

실제 모델 전체 autoregressive decoding 기준 측정. `bench_decode.py` + `torch.compile(fullgraph=True)` 사용.

### 실험 환경

| 항목 | 값 |
|------|----|
| GPU | NVIDIA A100 (80GB) |
| 세팅 | E8-D4, r=1.75, auto_mask, spct=5%, n_samples=128 |
| 측정 방식 | 4개 프롬프트 × 256 tokens, compile+fullgraph |

### 결과 (tok/s)

| 모델 | FP16 decode | EPTQ CUDA | 가속비 |
|---|---|---|---|
| Llama-2-7b-hf | ~50 | **235** | ~4.7× |
| Llama-2-13b-hf | ~30 | **142** | ~4.7× |
| Meta-Llama-3-8B | ~45 | **193** | ~4.3× |
| Llama-3.1-8B-Instruct | ~45 | **191** | ~4.2× |
| Llama-2-70b-hf | ~9* | **36** | ~4× |

> *70b FP16은 단일 A100에서 KV-cache 포함 barely fit; 실용적 FP16 단일 GPU decode 불가.

### spct 민감도 (Llama-2-7b-hf, A100)

| spct | tok/s |
|---|---|
| 1% | 234 |
| 2% | 235 |
| 5% | 235 |
| 10% | 227 |
| 50% | 229 |

Critical weight scatter 오버헤드는 spct=50%에서도 ±3% 이내로 무시 가능.

### QTIP 비교 (Llama-2-70b, A100)

| 방식 | tok/s |
|---|---|
| EPTQ E8D4 (CUDA) | 36 |
| QTIP (보고 수치) | ~39 |

약 8% 차이. 원인: QTIP은 critical weight scatter 없이 순수 VQ decode. 8B 이하 모델에서는 EPTQ가 동등하거나 빠름.

---

## Layer-level 마이크로 벤치마크 (RTX 4080)

CUDA inference kernel (`EPTQLinear`)과 fp16 복원 후 추론(`reconstruct_model`)의 레이어 단위 속도를 비교합니다.

### 실험 환경

| 항목 | 값 |
|------|----|
| GPU | NVIDIA GeForce RTX 4080 |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.1 |
| 측정 방식 | 50회 반복 중앙값, CUDA synchronize 기준 |
| 입력 dtype | fp16 |

### 비교 대상 정의

| 방식 | 설명 |
|------|------|
| **Reconstruct** | `dequantize_layer()` 로 fp16 weight를 미리 VRAM에 올린 뒤 `torch.mm` |
| **Kernel** | `EPTQLinear.forward()` — 양자화 포맷을 CUDA 커널이 on-the-fly 복원하며 matmul |

Reconstruct는 복원 시간을 측정에 포함하지 않습니다 (한 번 로드 후 반복 추론 가정).

---

### E8 Lattice 결과

| 레이어 (out × in) | Batch | Reconstruct | Kernel | Speedup |
|-------------------|-------|-------------|--------|---------|
| OPT-125M attn (768×768) | 1 | 0.022 ms | 0.047 ms | 0.47× |
| OPT-125M attn (768×768) | 8 | 0.023 ms | 0.047 ms | 0.49× |
| OPT-125M attn (768×768) | 32 | 0.023 ms | 0.065 ms | 0.35× |
| OPT-125M FFN (3072×768) | 1 | 0.023 ms | 0.047 ms | 0.48× |
| OPT-125M FFN (3072×768) | 8 | 0.023 ms | 0.060 ms | 0.38× |
| OPT-125M FFN (3072×768) | 32 | 0.022 ms | 0.136 ms | 0.16× |
| LLaMA-7B attn (4096×4096) | 1 | 0.030 ms | 0.062 ms | 0.48× |
| LLaMA-7B attn (4096×4096) | 8 | 0.030 ms | 0.234 ms | 0.13× |
| LLaMA-7B attn (4096×4096) | 32 | 0.036 ms | 0.793 ms | 0.05× |
| LLaMA-7B FFN gate/up (11008×4096) | 1 | 0.137 ms | 0.100 ms | **1.38×** |
| LLaMA-7B FFN gate/up (11008×4096) | 8 | 0.154 ms | 0.584 ms | 0.26× |
| LLaMA-7B FFN gate/up (11008×4096) | 32 | 0.196 ms | 2.135 ms | 0.09× |
| LLaMA-7B FFN down (4096×11008) | 1 | 0.136 ms | 0.117 ms | **1.16×** |
| LLaMA-7B FFN down (4096×11008) | 8 | 0.156 ms | 0.571 ms | 0.27× |
| LLaMA-7B FFN down (4096×11008) | 32 | 0.177 ms | 2.029 ms | 0.09× |

### E8-D4 결과

| 레이어 (out × in) | Batch | Reconstruct | Kernel | Speedup |
|-------------------|-------|-------------|--------|---------|
| OPT-125M attn (768×768) | 1 | 0.021 ms | 0.046 ms | 0.46× |
| OPT-125M attn (768×768) | 8 | 0.022 ms | 0.047 ms | 0.48× |
| OPT-125M attn (768×768) | 32 | 0.022 ms | 0.054 ms | 0.41× |
| OPT-125M FFN (3072×768) | 1 | 0.022 ms | 0.046 ms | 0.48× |
| OPT-125M FFN (3072×768) | 8 | 0.023 ms | 0.055 ms | 0.41× |
| OPT-125M FFN (3072×768) | 32 | 0.025 ms | 0.112 ms | 0.22× |
| LLaMA-7B attn (4096×4096) | 1 | 0.034 ms | 0.050 ms | 0.67× |
| LLaMA-7B attn (4096×4096) | 8 | 0.034 ms | 0.106 ms | 0.32× |
| LLaMA-7B attn (4096×4096) | 32 | 0.038 ms | 0.304 ms | 0.13× |
| LLaMA-7B FFN gate/up (11008×4096) | 1 | 0.137 ms | 0.060 ms | **2.28×** |
| LLaMA-7B FFN gate/up (11008×4096) | 8 | 0.154 ms | 0.216 ms | 0.71× |
| LLaMA-7B FFN gate/up (11008×4096) | 32 | 0.202 ms | 0.742 ms | 0.27× |
| LLaMA-7B FFN down (4096×11008) | 1 | 0.137 ms | 0.062 ms | **2.23×** |
| LLaMA-7B FFN down (4096×11008) | 8 | 0.156 ms | 0.204 ms | 0.77× |
| LLaMA-7B FFN down (4096×11008) | 32 | 0.179 ms | 0.677 ms | 0.26× |

---

### 메모리 사용량

양자화 방식에 무관하게 동일한 압축률:

| 레이어 | fp16 weight | 양자화 weight | 절약 |
|--------|-------------|--------------|------|
| OPT-125M attn (768×768) | 1.18 MB | 0.15 MB | 87.0% |
| OPT-125M FFN (3072×768) | 4.72 MB | 0.61 MB | 87.1% |
| LLaMA-7B attn (4096×4096) | 33.55 MB | 4.28 MB | 87.3% |
| LLaMA-7B FFN (11008×4096) | 90.18 MB | 11.48 MB | 87.3% |

---

### 분석

#### Batch=1에서 대형 레이어의 커널 우위

Batch=1 autoregressive decoding은 **메모리 대역폭 병목**입니다. GPU 연산 유닛이 weight를 HBM에서 읽어오는 속도에 의해 제한됩니다.

커널은 int16 인덱스(2 bytes)를 읽어 코드북에서 8개 fp16 값을 복원하므로,  
실질적으로 HBM에서 읽는 weight 데이터가 약 **8배 감소**합니다.  
이 대역폭 이득이 코드북 조회 오버헤드를 상쇄하면 커널이 더 빨라집니다.

LLaMA-7B FFN (11008×4096) 기준:
- E8 커널: **1.38×** 빠름
- E8-D4 커널: **2.28×** 빠름 (코드북 크기가 4 KB로 L1 캐시에 완전히 상주)

E8-D4가 E8보다 빠른 이유: E8 코드북은 ~1 MB(L2 상주)인 반면, E8-D4 코드북은 ~4 KB로 **L1 캐시에 완전히 올라가** 랜덤 접근 비용이 없습니다.

#### Batch≥8에서 Reconstruct 우위

배치가 커질수록 연산이 compute-bound로 전환됩니다. cuBLAS의 고도로 최적화된 GEMM(Tensor Core 활용)과 달리, 현재 커널은 코드북 기반 복원 방식으로 인해 Tensor Core를 사용하지 못합니다. 결과적으로 Batch=32에서 커널이 fp16 matmul 대비 3~11배 느려집니다.

#### 소형 모델에서 커널이 느린 이유

OPT-125M 수준의 작은 레이어에서는 GPU가 어떤 방식이든 idle 시간이 많습니다. 커널 런치 오버헤드와 코드북 조회 지연이 두드러지며, cuBLAS의 최적화된 경로가 절대적으로 유리합니다.

---

### 사용 가이드라인

| 상황 | 권장 방식 | 이유 |
|------|-----------|------|
| Single-token decoding, 대형 모델 (≥7B) | **CUDA kernel** | 메모리 대역폭 이득 |
| Single-token decoding, 소형 모델 (~125M) | Reconstruct | 커널 오버헤드가 이득 초과 |
| Prefill (batch ≥ 8) | **Reconstruct** | cuBLAS가 압도적으로 빠름 |
| VRAM 부족 상황 | **CUDA kernel** | 87% 메모리 절약 (배치 무관) |

---

### 한계 및 향후 개선 방향

- **Batch=1 소형 레이어**: 커널 오버헤드(코드북 lookup, warp reduction)를 줄이는 최적화 필요
- **대형 배치 지원**: Tensor Core를 활용하는 dequant-fused GEMM 커널 구현 필요
- **FP8 활용**: 현재 fp32 중간 누산기를 FP8/BF16으로 교체하면 메모리 압력 감소 가능
- **Persistent kernel**: 여러 레이어를 하나의 커널 실행으로 처리하면 커널 런치 오버헤드 절감 가능
