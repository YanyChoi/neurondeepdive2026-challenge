# NDD Day-2 Challenge — Qwen3-Embedding-8B를 vLLM Neuron에 온보딩하기

AWS Neuron Deep Dive Day-2 챌린지 과제: **Qwen/Qwen3-Embedding-8B**를
vLLM Neuron Plugin(release-0.24)에 직접 온보딩하고, trn2.3xlarge(4 NeuronCores)에서
검증/벤치마크합니다.

- 온보딩 가이드: [awslabs accelerated-compute-tutorials — vLLM model onboarding](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding/)
- 공식 레시피: [Neuron Docs — Qwen3-Embedding recipe](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-recipes/qwen3-embedding-8b.html)

## 레포 구조

```
step0_arch_diff.py            # Step 0: Llama vs Qwen3-Embedding 아키텍처 diff
configs/                      # 비교에 쓰는 두 모델의 config.json
src/qwen3_embedding/          # Stage 1: llama3/ 템플릿 → Qwen3-Embedding 포팅
  ├── config.py               #   HF config → dataclass (8B 파라미터)
  ├── model.py                #   백본 (QK-norm, 표준 RoPE 등 수정 적용)
  ├── model_embedding.py      #   pooling 러너용 임베딩 모델 (lm_head 제거)
  └── factory.py              #   러너 타입에 따라 구현 선택 + 등록용 팩토리
scripts/
  ├── setup_env.sh            # DLC 컨테이너 or 소스 설치 (release-0.24)
  ├── install_into_plugin.py  # Stage 2: 플러그인에 모델 설치 + registry 패치
  ├── launch_server.sh        # Stage 3: vllm serve (--runner pooling, TP=4)
  ├── smoke_test.py           # Stage 3: /v1/embeddings 스모크 테스트
  ├── offline_embed.py        # Stage 3: LLM.embed() 오프라인 경로
  ├── eval_mteb.py            # Stage 4: MTEB STS12/NFCorpus/SciFact
  └── benchmark.py            # Stage 5: throughput/latency 측정
```

## Step 0 — 아키텍처 Diff 분석

```bash
python3 step0_arch_diff.py
```

config.json 필드 비교 + 체크포인트 구조 비교 결과 요약:

| 항목 | Llama-3.1-8B | Qwen3-Embedding-8B | 코드에 미치는 영향 |
|---|---|---|---|
| layers / hidden / interm | 32 / 4096 / 14336 | **36** / 4096 / **12288** | config.py 수치만 |
| heads (Q/KV, head_dim) | 32 / 8, 128 | 32 / 8, 128 (명시적) | 동일 — GQA/TP 코드 재사용 |
| vocab | 128256 | **151665** | config.py |
| **QK-norm** | 없음 | **q_norm/k_norm (per-head, head_dim=128)** | **model.py 핵심 수정** — prefill은 RoPE 전에 per-head RMSNorm, decode는 megakernel `rmsnorm_QK_pre_rope_enabled=True`, weight mapping 2개 추가(TP 샤딩 없음) |
| RoPE | theta 5e5 + llama3 rope_scaling | theta **1e6**, scaling **없음** | Llama3 piecewise scaling 코드 제거, 표준 rotate_half |
| rms_norm_eps | 1e-5 | **1e-6** | config.py |
| lm_head | 있음 | **없음** (임베딩 체크포인트) | model_embedding.py — lm_head 제거, weight key도 `model.` prefix 없음 |
| 출력 | logits → 샘플링 | **[T,H] hidden → LAST pooling → L2 normalize** | forward가 hidden states 반환, vLLM pooling 러너(DispatchPooler)가 처리. **prefill-only** (decode 그래프 없음) |
| architectures | LlamaForCausalLM | **Qwen3ForCausalLM** (임베딩도 동일!) | 등록 키는 `Qwen3ForCausalLM`, 임베딩 여부는 `--runner pooling`이 결정 |

## Stage 1–2 — Implement + Register

`src/qwen3_embedding/`은 vllm-neuron의 `vllm_neuron/model/llama3/` 템플릿을 복사해
위 diff를 적용한 포팅입니다 (upstream vllm-neuron은 Apache-2.0; 파일 헤더에
수정 내역 주석). 등록은 온보딩 가이드대로 `vllm_neuron/model/registry.py`를 통합니다 —
`NeuronWorker.__init__`이 worker 프로세스마다 이 registry로 vLLM ModelRegistry를
덮어쓰기 때문에, 단순 `ModelRegistry.register_model()`만으로는 부족합니다.

```bash
# 서빙 환경(컨테이너/venv) 안에서:
python3 scripts/install_into_plugin.py
# [install] verified: Qwen3ForCausalLM -> vllm_neuron.model.qwen3_embedding...
```

## Stage 3 — Compile & Smoke Test (trn2.3xlarge)

```bash
./scripts/setup_env.sh venv          # 또는: ./scripts/setup_env.sh dlc <image-uri>
python3 scripts/install_into_plugin.py
./scripts/launch_server.sh           # TP=4, --runner pooling, 최초 기동 시 NEFF 컴파일
python3 scripts/smoke_test.py        # health / 4096-dim / L2 norm / 의미 순위 검증
```

서버 로그에 `[NDD-D2-CHALLENGE] Using challenge Qwen3ForEmbedding ...`이 찍히면
빌트인 구현이 아닌 이 레포의 구현이 실행되고 있는 것입니다.

임베딩 모델은 decode 단계가 없어 prefill 그래프만 컴파일됩니다
(`num_batched_tokens_buckets`가 컴파일되는 shape을 결정).

## Stage 4 — Accuracy (MTEB)

```bash
pip install "mteb>=1.25" numpy requests
python3 scripts/eval_mteb.py
```

쿼리에는 공식 Qwen3 instruction 템플릿(`Instruct: {task}\nQuery: {text}`)을 적용하고
문서는 그대로 임베딩합니다. 기준값:

| Task | Metric | 공식 MTEB (published) | Neuron BF16 (공식 레시피) | 본 구현 |
|---|---|---|---|---|
| STS12 | Spearman | 0.8614 | 0.8639 | _(측정 예정)_ |
| NFCorpus | NDCG@10 | 0.4145 | 0.4143 | _(측정 예정)_ |
| SciFact | NDCG@10 | 0.7846 | 0.7839 | _(측정 예정)_ |

## Stage 5 — Benchmark

```bash
python3 scripts/benchmark.py --concurrency 8 --batch-size 8 --num-requests 200
```

requests/sec, embeddings/sec, latency p50/p95/p99를 `results/`에 저장합니다.

## 메모

- `--runner pooling` 필수: 체크포인트의 architectures가 `Qwen3ForCausalLM`이라
  플래그 없이는 생성 모델로 로드됩니다.
- 짧은 요청 위주 임베딩 워크로드에는 `--no-enable-prefix-caching`이 유리
  (segmented prefill 오버헤드 회피).
- 출력 벡터는 L2-normalized — 클라이언트에서 dot product가 곧 cosine similarity.
- Matryoshka(차원 축소)를 쓰려면 `--hf-overrides '{"is_matryoshka": true}'`로 기동
  후 요청에 `dimensions` 지정.
