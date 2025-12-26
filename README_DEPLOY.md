
Deployment guide — ONNX quantization (CPU) and Vertex AI (GPU)

This document summarizes deployment options and practical commands to:
- Export and quantize a Hugging Face causal LM to ONNX and serve on Cloud Run (CPU path)
- Deploy a GPU-backed inference endpoint using Vertex AI (recommended for sub-200ms latency)

Prerequisites
- gcloud CLI configured and authenticated
- Docker installed
- Python 3.8+ for export/quantize steps

1) Artifact Registry + Cloud Run (quick setup)

Enable required APIs:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

Create an Artifact Registry (one-time):

```bash
gcloud artifacts repositories create chatbot-repo --repository-format=docker --location=us-central1
```

Build & deploy helper (already in repository):

```bash
# set env vars then
./scripts/deploy.sh
```

4) After deployment, note the service URL from gcloud output and test /chat endpoint.

2) CPU path: ONNX export + quantization (best-effort to approach <200ms)

Export an ONNX model suitable for CPU inference. Note: exporting full causal LM with past_key_values requires model-specific graph adjustments.

```bash
python src/onnx_export_optimized.py --model outputs/distilgpt2-dialogue --out onnx
```

This writes `onnx/model.onnx`.

Quantize the ONNX model to int8 (post-training dynamic quantization):

```bash
python src/quantize_onnx.py --input onnx/model.onnx --output onnx/model.quant.onnx
```

Serve with ONNX Runtime (CPU-optimized): build the ONNX Docker image and deploy to Cloud Run. Use `--concurrency=1` and `--min-instances=1` to reduce per-request latency.

```bash
# build using Dockerfile.onnx
docker build -f Dockerfile.onnx -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/chatbot-onnx:latest .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/chatbot-onnx:latest

gcloud run deploy onnx-chatbot --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/chatbot-onnx:latest --platform managed --region ${REGION} --concurrency=1 --min-instances=1 --memory=2Gi --allow-unauthenticated
```

Notes about CPU path and expectations:
- Quantization + ONNX Runtime can deliver meaningful speedups (2-4x) especially on CPUs with AVX/oneDNN support, but may still be above 200ms for medium-sized models.
- Use `onnxruntime` builds that enable MKL/oneDNN and OpenMP for maximum throughput.
- For small models and short responses you stand a better chance of approaching 200ms on high-CPU Cloud Run instances.

3) GPU path: Vertex AI (recommended for strict <200ms)

Vertex AI provides managed GPUs for inference. Steps summary:

1. Upload model to GCS

```bash
gsutil cp -r outputs/distilgpt2-dialogue gs://$BUCKET/models/distilgpt2-dialogue
```

2. Create Vertex AI model resource

```bash
gcloud ai models upload --region=$REGION --display-name=distilgpt2-dialogue --artifact-uri=gs://$BUCKET/models/distilgpt2-dialogue
```

3. Create endpoint and deploy with GPU

```bash
gcloud ai endpoints create --region=$REGION --display-name=chatbot-endpoint
# Deploy model: replace ENDPOINT_ID and MODEL_ID after create/upload
gcloud ai endpoints deploy-model ENDPOINT_ID --model=MODEL_ID --region=$REGION --machine-type=n1-standard-8 --accelerator=count=1,type=nvidia-tesla-t4
```

Notes:
- Vertex AI gives you GPU-backed inference with optimized runtimes and lower latency; this is the most reliable path for <200ms for non-trivial models.
- Costs are higher, but latency and throughput are typically far better than CPU-only Cloud Run.

Performance checklist
- Run `scripts/measure_latency.py` locally against the running service and against the deployed URL.
- Measure p50/p95 latency and throughput with realistic prompts.
- Tune: model size, max_new_tokens, quantization, CPU affinity, min-instances, memory/cpu and concurrency.

Recommended next steps
- If you want me to implement the full ONNX token-by-token decoding loop and a production-ready ONNX serving image (further reducing CPU latency), say "Implement ONNX decoding" and I'll add the code and Dockerfile changes.
- If you want Vertex AI deployment automation (model upload and endpoint creation), say "Vertex AI deploy" and I'll create scripts and sample commands.
