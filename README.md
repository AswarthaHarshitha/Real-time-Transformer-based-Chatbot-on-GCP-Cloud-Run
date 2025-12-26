## 🚀 Deployment Guide — ONNX (CPU) and Vertex AI (GPU)

This project supports two deployment paths:
- Cloud Run (CPU) using ONNX + quantization
- Vertex AI (GPU) for low-latency production (recommended)

-------------------------------------
✅ Prerequisites
-------------------------------------
- Python 3.8+
- Docker installed
- Google Cloud gcloud CLI installed and authenticated

Run:

gcloud auth login
gcloud config set project YOUR_PROJECT_ID

-------------------------------------
1️⃣ Quick Deployment — Artifact Registry + Cloud Run
-------------------------------------

Enable APIs:

gcloud services enable run.googleapis.com artifactregistry.googleapis.com

Create Artifact Registry:

gcloud artifacts repositories create chatbot-repo \
  --repository-format=docker \
  --location=us-central1

Deploy using script:

./scripts/deploy.sh

After deployment:
- copy the Cloud Run URL
- test the /chat endpoint

-------------------------------------
2️⃣ CPU Path — ONNX Export + Quantization
-------------------------------------

This is cost-efficient and suitable for moderate latency workloads.

Step 1 — Export model to ONNX

python src/onnx_export_optimized.py \
  --model outputs/distilgpt2-dialogue \
  --out onnx

Output generated:
onnx/model.onnx

Step 2 — Quantize model to INT8

python src/quantize_onnx.py \
  --input onnx/model.onnx \
  --output onnx/model.quant.onnx

Step 3 — Build and deploy ONNX container

Build image:

docker build -f Dockerfile.onnx \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/chatbot-onnx:latest .

Push image:

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/chatbot-onnx:latest

Deploy to Cloud Run:

gcloud run deploy onnx-chatbot \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/chatbot-onnx:latest \
  --platform managed \
  --region ${REGION} \
  --concurrency=1 \
  --min-instances=1 \
  --memory=2Gi \
  --allow-unauthenticated

CPU performance notes:
- ONNX + INT8 can give 2–4x speedup
- <200ms depends on small models and short outputs
- best results when ONNX Runtime uses oneDNN / MKL

-------------------------------------
3️⃣ GPU Path — Vertex AI (Recommended)
-------------------------------------

Use this when you need:
- <200ms latency
- larger models
- production stability

Step 1 — Upload model to Cloud Storage

gsutil cp -r outputs/distilgpt2-dialogue gs://$BUCKET/models/distilgpt2-dialogue

Step 2 — Register model in Vertex AI

gcloud ai models upload \
  --region=$REGION \
  --display-name=distilgpt2-dialogue \
  --artifact-uri=gs://$BUCKET/models/distilgpt2-dialogue

Step 3 — Create endpoint and deploy to GPU

Create endpoint:

gcloud ai endpoints create \
  --region=$REGION \
  --display-name=chatbot-endpoint

Deploy model (replace ENDPOINT_ID and MODEL_ID):

gcloud ai endpoints deploy-model ENDPOINT_ID \
  --model=MODEL_ID \
  --region=$REGION \
  --machine-type=n1-standard-8 \
  --accelerator=count=1,type=nvidia-tesla-t4

-------------------------------------
📊 Performance Checklist
-------------------------------------

Measure latency:

python scripts/measure_latency.py

Test against:
- local server
- deployed URL

Tune:
- model size
- max_new_tokens
- quantization level
- CPU/memory tier
- number of workers
- concurrency
- min-instances warm start

-------------------------------------
✅ Summary
-------------------------------------
Cloud Run + ONNX (CPU): cheaper and simpler, good for prototypes
Vertex AI (GPU): lowest latency, best for production chat systems
