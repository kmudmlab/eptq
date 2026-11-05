# EPTQ: E8 Lattice Post-Training Quantization

E8 Lattice 기반의 LLM Post-Training Quantization 라이브러리입니다.

## Features

- **E8 Lattice Quantization**: 8차원 E8 격자 구조를 활용한 효율적인 벡터 양자화
- **E8-D4 Quantization**: D4 격자 기반의 대안적 양자화 방식 지원
- **Critical Weight Masking**: 중요 가중치를 원본 정밀도로 보존하여 정확도 유지
- **Auto Mask (Kneedle)**: 레이어별 RTN D_sum 분포에서 자동으로 임계값 탐색
- **Layer-wise Quantization**: 메모리 효율적인 레이어별 양자화 및 저장
- **CUDA Inference Kernel**: fp16 복원 없이 양자화 포맷에서 직접 추론 (메모리 87% 절약)
- **지원 모델**: OPT, LLaMA 계열

## Installation

```bash
pip install -e .
```

## Usage

### 1. Quantization

모델을 양자화하여 저장합니다:

```bash
python -m eptq.quantize <model_name> --output_dir <output_dir> [options]
```

**예시:**
```bash
# E8-D4, auto_mask (권장)
python -m eptq.quantize meta-llama/Llama-2-7b-hf \
    --output_dir ./quantized_llama2-7b \
    --quantizer e8_d4 --codebook_radius 1.75 \
    --auto_mask --auto_mask_search_pct 0.1

# E8 Lattice (기본값), fixed percmask
python -m eptq.quantize facebook/opt-125m \
    --output_dir ./quantized_opt-125m --percmask 0.002
```

**주요 옵션:**
| Option | Default | Description |
|--------|---------|-------------|
| `--quantizer` | `e8_lattice` | 양자화 방식 (`e8_lattice` 또는 `e8_d4`) |
| `--codebook_radius` | None | 코드북 반경 (미지정 시: e8_lattice=1.6, e8_d4=1.875) |
| `--percdamp` | 0.01 | Hessian damping 비율 |
| `--percmask` | 0.002 | Critical weight 비율 (top-k blocks, 전역 고정) |
| `--mask_tau` | None | RTN D_sum 임계값으로 critical weight 결정 (레이어별 가변). `--percmask` 대체 |
| `--auto_mask` | False | Kneedle 알고리즘으로 레이어별 임계값 자동 탐색. `--percmask`, `--mask_tau` 대체 |
| `--auto_mask_search_pct` | 0.10 | `--auto_mask` 탐색 윈도우 상위 X% |
| `--radius_candidates` | None | 레이어별 최적 반경 탐색 (예: `1.7,1.75,1.8`). 손실 최소 후보 선택 |
| `--no_weight_scale_norm` | False | Weight scale normalization (g/h) 비활성화 |
| `--qb_size` | None | Row group 크기 (group quantization). None = per-column scale |
| `--calib_dataset` | `c4` | Calibration 데이터셋 (`c4` 또는 `wikitext2`) |
| `--n_samples` | 128 | Calibration 샘플 수 |
| `--seqlen` | 2048 | Calibration 시퀀스 길이 |
| `--blocksize` | 128 | GPTQ-style 양자화 블록 크기 |
| `--seed` | 42 | Random seed |
| `--fp16_hessian` | False | Quant chain + FP16 chain Hessian 합산 (H = H_quant + H_fp16). Effective calibration samples 2배 |
| `--device` | `cuda` | 사용할 디바이스 |

**Proxy Fine-tuning 옵션 (`--ft_epochs > 0` 시 활성화):**
| Option | Default | Description |
|--------|---------|-------------|
| `--ft_epochs` | 0 | 레이어별 proxy fine-tuning epoch 수 (0 = 비활성화) |
| `--ft_lr` | 1e-4 | Adam learning rate |
| `--ft_batchsize` | 8 | Fine-tuning batch size |
| `--ft_valid_size` | 0 | Early stopping용 validation 샘플 수 (0 = 비활성화) |
| `--ft_early_stop` | 5 | Validation loss 개선 없을 시 조기 종료 patience |
| `--ft_tune_O` | False | Critical weight 값 O도 함께 fine-tuning |
| `--ft_loss` | `mse` | Fine-tuning loss (`mse`, `cosine`, `mse+cosine`) |
| `--ft_fp16_chain` | False | Fine-tuning target을 FP16 chain 출력으로 설정 (기본: quantized chain 입력 위의 FP16 출력) |
| `--ft_e2e` | False | 전체 레이어 양자화 후 end-to-end CE loss fine-tuning 실행 |

### 2. Reconstruction

양자화된 모델을 fp16으로 복원합니다:

```bash
python -m eptq.reconstruct --quantized_dir <quantized_dir> --output_dir <output_dir>
```

### 3. Direct Inference (CUDA Kernel)

fp16으로 복원하지 않고 양자화된 포맷에서 직접 추론합니다. 가중치를 int16 인덱스로 유지하므로 **메모리 사용량이 약 87% 줄어듭니다**.

```bash
# CUDA 커널 사전 빌드 (선택사항 — 없으면 첫 실행 시 JIT 컴파일)
python3 setup_cuda.py build_ext --inplace
```

```python
from eptq.kernels.patch import load_quantized_model

model, tokenizer = load_quantized_model("./quantized_opt-125m", device="cuda")

inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

> 구현 상세 및 성능 비교는 [docs/kernel_inference.md](docs/kernel_inference.md)와 [docs/benchmark_results.md](docs/benchmark_results.md)를 참고하세요.

### 4. Evaluation

PPL, zero-shot, 추론 속도를 평가합니다:

```bash
# PPL 평가 (wikitext2, ptb, c4)
python eval_ppl.py --quantized_dir ./quantized_opt-125m --dataset wikitext2 ptb c4

# Zero-shot lm-eval (5개 태스크)
python eval_zero_shot.py --quantized_dir ./quantized_opt-125m \
    --tasks piqa,hellaswag,arc_easy,arc_challenge,winogrande \
    --batch_size auto --output_path ./results.json

# Decode 속도 벤치마크 (CUDA 커널)
python3 setup_cuda.py build_ext --inplace
EPTQ_CUDA_DECODE=1 python bench_decode.py \
    --model_dir ./quantized_opt-125m --quantized --compile --fullgraph \
    --label my_exp --results_file ./bench.json
```

## Output Structure

양자화 후 생성되는 파일 구조:

```
quantized_model/
├── quant_config.json          # 양자화 설정
├── config.json                # HuggingFace 모델 설정
├── quantizer.pt               # Quantizer codebook (E8 Lattice 또는 E8-D4)
├── non_quantized_params.pt    # 글로벌 비양자화 파라미터 (embeddings, final_norm, lm_head)
├── tokenizer/                 # Tokenizer 파일들
│   ├── tokenizer.json
│   ├── vocab.json
│   └── ...
├── layer_0/
│   ├── self_attn.k_proj.pt    # 양자화된 Linear 레이어
│   ├── self_attn.v_proj.pt
│   ├── self_attn.q_proj.pt
│   ├── self_attn.out_proj.pt
│   ├── fc1.pt
│   ├── fc2.pt
│   └── non_quantized.pt       # 레이어별 비양자화 파라미터 (layer norms)
├── layer_1/
│   └── ...
└── ...
```

## Quantized Data Format

각 양자화된 Linear 레이어 (`*.pt`)는 `QuantizedLayerData`로 저장되며 다음 필드를 포함합니다:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `Q` | int16 | (out//8, in) | 양자화 인덱스 (32767 = critical weight 마커) |
| `g` | fp16 | (out,) | Row scale factors |
| `h` | fp16 | (in,) | Column scale factors (column scale S 흡수됨) |
| `O` | fp16 | (n_masked, 8) | Critical weight 원본 값 (g/h 정규화 공간) |
| `bias` | fp16 | (out,) | Bias (optional) |
| `chosen_r` | float | — | 이 레이어에 사용된 codebook 반경 |
| `S_group` | fp16 | (n_groups, in) | Group quantization scale (qb_size 사용 시만 존재) |

> `Q` 값 해석: 저장 시 `Q_int16 = Q_uint16 - 32768`. Critical block은 원래 인덱스 65535 → int16으로 32767.

## Project Structure

```
eptq/
├── __init__.py
├── config.py              # QuantizationConfig, QuantizedLayerData
├── quantize.py            # 양자화 메인 스크립트
├── reconstruct.py         # fp16 복원
├── eval_ppl.py            # PPL 평가 (wikitext2, ptb, c4)
├── eval_zero_shot.py      # Zero-shot lm-eval
├── bench_decode.py        # Decode 속도 벤치마크
├── algorithms/
│   └── eptq.py            # EPTQ 알고리즘 (comp_gh, eptq_quantize, dequantize_layer)
├── models/
│   ├── base.py            # ModelHandler 추상 클래스
│   ├── opt.py             # OPT handler
│   └── llama.py           # LLaMA handler
├── quantizers/
│   ├── e8_lattice.py      # E8 Lattice quantizer
│   └── e8_d4.py           # E8-D4 quantizer
├── kernels/               # CUDA inference kernel
│   ├── __init__.py
│   ├── linear.py          # EPTQLinear (nn.Linear 대체)
│   └── patch.py           # load_quantized_model()
├── csrc/                  # CUDA / C++ 소스
│   ├── eptq_kernels.h
│   ├── eptq_kernels.cu
│   └── binding.cpp
├── setup_cuda.py          # CUDA 확장 빌드 스크립트
├── docs/
│   ├── kernel_inference.md
│   └── benchmark_results.md
└── utils/
    ├── calibration.py     # Calibration data loading
    └── hessian.py         # Hessian computation
```

## Algorithm Overview

EPTQ는 다음 단계로 동작합니다:

1. **Calibration**: C4 데이터셋에서 calibration 샘플 수집
2. **Hessian 계산**: 각 레이어의 입력으로부터 Hessian 행렬 계산 (`--fp16_hessian` 시 quant+FP16 chain 합산)
3. **Weight Scale Normalization**: `comp_gh()`로 row/column 스케일 인수 g, h 계산 후 정규화
4. **Critical Weight Selection**: RTN pre-pass로 8D 블록별 D_sum 계산 → top-k (`--percmask`), 임계값 (`--mask_tau`), 또는 Kneedle 자동 탐색 (`--auto_mask`) 중 택일
5. **Vector Quantization**: 8차원 벡터 단위 E8/E8-D4 양자화, GPTQ-style 오류 전파
6. **Critical Weight Preservation**: 선택된 블록은 `Q[i,j] = 65535`로 마킹, 원본 값은 `O`에 fp16으로 보존
7. **Proxy Fine-tuning** (optional): g, h scale을 레이어 출력 MSE 기준으로 fine-tuning

## Supported Models

- `facebook/opt-*` (OPT 계열)
- `meta-llama/Llama-*` (LLaMA 계열)

새로운 모델을 지원하려면 `eptq/models/`에 `ModelHandler` 서브클래스를 구현하세요. LLaMA와 구조가 유사한 모델(예: Qwen)은 `LlamaHandler`를 최소한으로 수정하여 추가할 수 있습니다.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers
- tqdm

## License

MIT License
