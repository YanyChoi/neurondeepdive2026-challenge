# NDD Day-2 Challenge — Qwen3-Embedding-8B를 vLLM Neuron에 온보딩하기

AWS Neuron Deep Dive Day-2 챌린지: **Qwen/Qwen3-Embedding-8B**를 vLLM Neuron
Plugin(release-0.24)에 직접 온보딩하고 trn2.3xlarge(4 NeuronCores)에서 검증합니다.
진행 순서는 [vLLM Neuron 모델 온보딩 가이드](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding/)의
5단계를 그대로 따랐습니다.

- 공식 레시피: [Neuron Docs — Qwen3-Embedding recipe](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-recipes/qwen3-embedding-8b.html)

## 온보딩의 본질: 2-Phase 접근

- **Phase 1 (필수)**: `NF.qkv_proj`, `NF.flash_attention`, `NF.o_proj` 등 Plugin
  빌딩블록으로 모델을 "교체" — NF 내부에 NKI 커널이 내장되어 있어 이것만으로
  NKI 최적화가 적용됩니다. **본 챌린지는 Phase 1까지** 수행했습니다.
- **Phase 2 (선택)**: 커스텀 NKI 커널 추가. 임베딩(prefill-only) 워크로드는 NF의
  flash attention 경로로 충분해 적용하지 않았습니다.

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. Implement    2. Register      3. Compile &    4. Validate    5. Benchmark  │
│ (config/model/   (ModelRegistry)   Smoke Test     Accuracy       & Tune       │
│  factory/weights)                                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Step 0: 아키텍처 Diff 분석

가이드의 공식 분석 스크립트 그대로 사용 (`MODEL_A`/`MODEL_B`만 지정):

```bash
pip install transformers torch
hf auth login                        # meta-llama/Llama-3.1-8B는 gated
python3 arch_diff_analysis.py        # 출력: results/step0_official_output.txt
```

### Diff 결과 해석 (가이드 체크리스트 기준)

| # | 비교 항목 | Llama-3.1-8B | Qwen3-Embedding-8B | 영향받는 코드 |
|---|---|---|---|---|
| 1 | Attention 방식 | 32Q/8KV GQA, head_dim 128, **QK-norm 없음**, sliding window 없음 | 32Q/8KV GQA, head_dim 128, **`q_norm`/`k_norm` per-head RMSNorm(dim=128)** — 구조 dump에서 발견된 최대 diff. `layer_types`=전부 full_attention | `model.py` — QK-norm을 RoPE **이전에** per-head 적용 (prefill: torch, decode: megakernel `rmsnorm_QK_pre_rope_enabled=True`), weight mapping 2개 추가(TP 샤딩 없음) |
| 2 | Heterogeneous Layers | 없음 — 32층 전부 동일 | 없음 — 36층 전부 동일 (full_attention ×36) | 레이어 수만 config 값 |
| 3 | Position Encoding | `rope_theta`=500,000 + **`rope_scaling`: llama3 piecewise (factor 8)** | `rope_theta`=**1,000,000**, `rope_scaling`=**None** (표준 rotate_half) | `model.py` — Llama3 스케일링 코드 삭제 |
| 4 | MLP/Activation | silu, `intermediate_size`=14336 | silu, `intermediate_size`=**12288** | 치수만 변경, NF 경로 호환 |
| 5 | Normalization | RMSNorm, `rms_norm_eps`=1e-5 | RMSNorm, `rms_norm_eps`=**1e-6** | `config.py` |
| 6 | Config 구조 | top-level | top-level (nested 없음) | `from_configs()` 그대로 |
| 7 | Embedding | `vocab_size`=128256, `tie_word_embeddings`=false | `vocab_size`=**151665**, `tie_word_embeddings`=false | embed_tokens 치수 |
| 8 | Special features | 일반 생성 체크포인트: `lm_head` 있음, weight key `model.` prefix, `architectures`=LlamaForCausalLM | **임베딩 체크포인트**: `lm_head` **없음**, weight key에 `model.` prefix **없음**, sentence-transformers 메타(modules.json: Transformer→LAST Pooling→Normalize), 그런데 `architectures`=**Qwen3ForCausalLM** (생성 모델과 동일) | `model_embedding.py` + `--runner pooling` |

## Stage 1: Implement (모델 구현)

가이드의 디렉토리 구조를 따랐습니다 (llama3 템플릿 복사 → 수정):

```text
src/model/qwen3_embedding/
├── __init__.py
├── config.py            # HF PretrainedConfig → dataclass, from_configs()
├── factory.py           # 러너 타입 검사 → 구현 선택, config 검증
├── model.py             # 백본: forward/get_kv_spec/bind_kv_cache/load_weights
├── model_embedding.py   # 임베딩 variant: lm_head 제거, [T,H] hidden 반환
└── register.py          # Stage 2 등록 스니펫
```

사용한 Building Blocks: `NF.qkv_proj`, `NF.flash_attention`, `NF.segmented_attention`,
`NF.o_proj`, `vllm_neuron.nn.VocabDimShardedEmbedding`, `fused_qkv_weight_loader`,
`sharding_weight_loader`, `KVSpec`/`LayerSpec`, `get_tp_group()` collectives.

임베딩 모델 특이사항: forward가 logits 대신 **flatten된 `[T, H]` post-norm hidden
states**를 반환하고, vLLM pooling 러너(DispatchPooler)가 LAST-token pooling + L2
normalize를 수행합니다. decode 단계가 없어 prefill 그래프만 컴파일됩니다.

## Stage 2: Register (모델 등록)

가이드의 등록 코드는 `src/model/qwen3_embedding/register.py`:

```python
from vllm import ModelRegistry
from .factory import Qwen3EmbeddingForCausalLM

ModelRegistry.register_model(
    "Qwen3ForCausalLM",   # ← HF config.json의 "architectures" 필드와 일치
    Qwen3EmbeddingForCausalLM,
)
```

단, Neuron plugin에서는 `NeuronWorker.__init__`이 worker 프로세스마다
`vllm_neuron.model.registry`로 vLLM ModelRegistry를 **다시 덮어쓰기** 때문에,
위 호출만으로는 built-in Qwen3 구현이 이깁니다. 그래서 설치 스크립트가 모델
디렉토리를 플러그인 안으로 복사하고 registry를 우리 구현으로 패치합니다:

```bash
python3 scripts/install_into_plugin.py
# [install] verified: Qwen3ForCausalLM -> vllm_neuron.model.qwen3_embedding...
```

## Stage 3: Compile & Smoke Test

가이드의 오프라인 스모크 (`LLM` — 임베딩 모델이므로 `generate` 대신 `embed`):

```bash
python3 scripts/offline_embed.py     # LLM(runner="pooling", TP=4) + llm.embed()
```

```
dim=4096  first3=[0.0204, -0.0047, 0.0026]  | Retrieval-augmented generation grounds a
dim=4096  first3=[0.013, 0.0038, -0.0003]   | def add(a, b):
dim=4096  first3=[0.0448, 0.0146, -0.0232]  | 당근마켓은 동네 이웃 간의 중고거래 플랫폼입니다.
dim=4096  first3=[-0.0113, 0.0226, 0.0198]  | Photosynthesis converts sunlight, water,
```

온라인 서버 + 스모크:

```bash
./scripts/launch_server.sh           # vllm serve --runner pooling, TP=4
python3 scripts/smoke_test.py
# [1/4] /health OK
# [2/4] 3 embeddings, dim=4096 OK
# [3/4] L2-normalized OK (norms=[1.0, 1.0, 1.0])
# [4/4] cos(query, Paris doc)=0.6442  cos(query, mitochondria doc)=0.1464
# SMOKE TEST PASSED
```

첫 실행에서 모든 bucket의 NEFF 컴파일이 이뤄지고(수 분~수십 분), 이후에는
compile cache를 재사용합니다. 서버 로그의 `[NDD-D2-CHALLENGE]` 마커로 built-in이
아닌 본 레포 구현이 실행됨을 확인했습니다.

## Stage 4: Validate Accuracy

가이드의 3-Level 검증 프레임워크를 임베딩 모델에 맞게 적용:

| Level | 가이드 | 본 챌린지 적용 | 결과 |
|---|---|---|---|
| L1: Task-level | lm_eval, longbench | **MTEB** (STS12/NFCorpus/SciFact — 챌린지 지정) | ✅ 아래 표 |
| L2: Prompt-level | teacher-forcing logit 3-way 비교 | **임베딩 벡터 3-way 비교** (Neuron vs HF FP32 vs HF BF16) | ✅ PASS |
| L3: Module-level | attention/MLP 단위 테스트 | 미수행 (L1+L2 통과로 생략) | — |

### L2: Three-Way Comparison

```bash
~/mteb-venv/bin/python scripts/accuracy_three_way.py
```

```
neuron_vs_fp32   max|Δ|=0.00369 mean|Δ|=0.000154 min_cos=0.999898
bf16_vs_fp32     max|Δ|=0.00296 mean|Δ|=0.000154 min_cos=0.999895
neuron_vs_bf16   max|Δ|=0.00338 mean|Δ|=0.000199 min_cos=0.999831
PASS: neuron_err=0.00369 vs bf16_err=0.00296 (allowed <= 2x + 1e-4)
```

해석 (가이드 기준): **Neuron 오차 ≈ BF16 오차 → 정상.** mean |Δ|는 소수 6자리까지
동일 — Neuron 실행의 수치 오차가 BF16 캐스팅 노이즈 수준 그 자체입니다.

### L1: MTEB (2026-08-24, trn2.3xlarge TP=4, BF16)

```bash
~/mteb-venv/bin/python scripts/eval_mteb.py
```

| Task | Metric | 공식 MTEB (published) | Neuron BF16 (공식 레시피) | **본 구현** |
|---|---|---|---|---|
| STS12 | Spearman | 0.8614 | 0.8639 | **0.8639** |
| NFCorpus | NDCG@10 | 0.4145 | 0.4143 | **0.4159** |
| SciFact | NDCG@10 | 0.7846 | 0.7839 | **0.7859** |

세 태스크 모두 published 대비 ±0.005 이내. 쿼리에는 공식 Qwen3 instruction
템플릿(`Instruct: {task}\nQuery: {text}`)을 적용, 문서는 그대로 임베딩.

## Stage 5: Benchmark & Tune

가이드의 벤치마크 방식 그대로 (`vllm bench serve`, 임베딩 백엔드):

```bash
./scripts/launch_server.sh
vllm bench serve --backend openai-embeddings --model Qwen/Qwen3-Embedding-8B \
    --dataset-name random --random-input-len 128 --num-prompts 200
```

**측정 결과** (request rate ∞, concurrency 무제한 — E2EL은 큐잉 포함):

| dataset | 요청 수 | Request throughput | Token throughput | Mean E2EL | P99 E2EL |
|---|---|---|---|---|---|
| random, input-len 128 | 200 | **38.67 req/s** | 4,950 tok/s | 2,637 ms | 5,106 ms |
| random, input-len 1024 | 100 | 13.57 req/s | **13,896 tok/s** | 3,737 ms | 7,283 ms |

입력이 길어질수록 토큰 처리량이 크게 오릅니다(4.9k → 13.9k tok/s) — prefill이
큰 배치 토큰 수에서 더 효율적이라는 뜻이고, 짧은 요청 구간의 ~38 req/s는 아래
closed-loop 측정의 ~38 emb/s 포화점과 일치합니다.
원본: [`results/vllm_bench_in128.json`](results/vllm_bench_in128.json),
[`results/vllm_bench_in1024.json`](results/vllm_bench_in1024.json)

보조 벤치마크 (`scripts/benchmark.py`, ~64단어 텍스트, closed-loop):

| concurrency × batch | embeddings/sec | p50 | p99 |
|---|---|---|---|
| 1 × 1 | 32.0 | 31.1 ms | 33.3 ms |
| 8 × 8 | **38.1** | 1620 ms | 1651 ms |
| 16 × 16 | 35.8 | 6487 ms | 6553 ms |

Tuning 적용 사항: `num_batched_tokens_buckets=[128..4096]` (prefill bucket),
`--no-enable-prefix-caching` (짧은 무공유 요청엔 segmented prefill 오버헤드 회피).
decode bucket / on-device sampling / FP8 KV cache는 임베딩(prefill-only) 모델에
해당 없음.

## 트러블슈팅 기록 (소스 설치 시)

공식 DLC 대신 소스로 설치하면 만나는 문제들 (전부 `scripts/setup_env.sh`에 반영됨):

1. **`torch-neuronx`를 설치하면 안 됩니다.** release-0.24는 `libtorch-neuronx-lite`
   (torch 2.11) 기반의 native PyTorch 스택입니다. pip repo의 torch-neuronx는 최신이
   2.9라서 설치하면 torch/torch-xla가 2.9로 다운그레이드되고 cu12/cu13 nvidia 라이브러리가
   충돌합니다 (`libtorch_cuda.so: undefined symbol: ncclDevCommDestroy`).
2. **`nki` 0.6.0 필수** (`nkilib`가 이 wheel에 포함). neuronx-cc 2.26은 nki 0.5를
   강제해서 `nkilib ... cannot import name 'CPCollectiveMode'`로 플러그인 로드가 깨집니다.
3. **`islpy==2026.1` 고정 필수.** islpy 2026.2.1(2026-08 릴리스)이 neuronx-cc 2.27의
   Simplifier를 깨뜨립니다:
   `[NCC_ISMP902] Simplifier error: is_subset(): incompatible function arguments`.
   서버 로그에는 `neuronx-cc compilation failed with 70`만 남고, 실제 원인은
   log-neuron-cc.txt가 아니라 **컴파일러 stdout**에만 찍히므로 로그의 neuronx-cc 명령을
   수동 재실행해야 보입니다. DLC는 빌드 시점 버전이 얼려져 있어 이 문제가 없습니다.
4. **trn2.3xlarge에는 EFA가 없습니다** → `NEURON_SKIP_EFA_AFFINITY=1`.
5. **버킷 규칙**: `num_batched_tokens_buckets`의 마지막 값 = `--max-num-batched-tokens`
   (vLLM 기본 2048). 튜토리얼의 `[128..4096]` 버킷은 `--max-num-batched-tokens 4096`을
   전제로 합니다.
6. **오프라인 `LLM()`도 APC를 꺼야 합니다**: single-shot prefill
   (`max_num_batched_tokens == max_model_len`)에서는 `enable_prefix_caching=False`
   없이는 기동이 거부됩니다.

검증 기준 버전 (DLC `vllm/inference/0.24.0.1.1.0/Dockerfile.neuronx` 고정값):
`neuronx-cc==2.27.5334.0+f702b353`, `nki==0.6.0+31049202112.g85070674`,
`libtorch-neuronx-lite==2.11.0.1.0.1284+f49d8626`, torch/torch-xla 2.11.0.

## 메모

- `--runner pooling` 필수: 체크포인트의 architectures가 `Qwen3ForCausalLM`이라
  플래그 없이는 생성 모델로 로드됩니다.
- 출력 벡터는 L2-normalized — 클라이언트에서 dot product가 곧 cosine similarity.
- Matryoshka(차원 축소)를 쓰려면 `--hf-overrides '{"is_matryoshka": true}'`로 기동
  후 요청에 `dimensions` 지정.
