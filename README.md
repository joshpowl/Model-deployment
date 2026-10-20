# Sentiment Classification API (FastAPI + Docker)

A small REST API that wraps the trained PyTorch sentiment MLP from the Deep Learning project. It uses the matching fitted TF-IDF vectorizer to turn review text into model features.

> **Model scope:** this is an educational binary classifier trained on a small example review dataset. Treat predictions as demonstrations, not production-grade sentiment analysis. The API does not download any model weights at runtime.

## Files

```text
Sentiment_Model_API_Project/
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI routes and model inference
├── artifacts/
│   ├── sentiment_mlp.pth      # Saved PyTorch weights + model metadata
│   └── tfidf_vectorizer.joblib # Matching fitted TF-IDF vectorizer
├── examples/
│   ├── request_positive.json
│   ├── request_negative.json
│   ├── sample_response.json
│   └── curl_demo.sh
├── tests/test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Option A — Run locally with Python

Use Python 3.11 recommended. Run commands from this project folder.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for interactive Swagger documentation. Leave the server running while sending requests from a second terminal.

## Option B — Run with Docker

Install Docker Desktop (Windows/macOS) or Docker Engine (Linux). From the project root:

```bash
docker build -t sentiment-api:1.0 .
docker run --rm -p 8000:8000 --name sentiment-api sentiment-api:1.0
```

Confirm the service is ready:

```bash
curl http://localhost:8000/health
```

Expected health response:

```json
{"status":"ok","model_loaded":true,"device":"cpu"}
```

Stop the container with `Ctrl+C` if running in the foreground, or use `docker stop sentiment-api` when running detached.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Basic service information |
| `GET` | `/health` | Health and model-loaded status |
| `POST` | `/predict` | Predict sentiment for one review |
| `GET` | `/docs` | Interactive OpenAPI/Swagger UI |

### Inference request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"The product is excellent and works perfectly."}'
```

Example response for the supplied trained artifact:

```json
{
  "text": "The product is excellent and works perfectly.",
  "predicted_sentiment": "positive",
  "positive_probability": 0.9596,
  "negative_probability": 0.0404,
  "threshold": 0.5
}
```

Probabilities can vary if the model artifact is retrained. Model probabilities are not necessarily calibrated confidence scores.

### Negative example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"This was disappointing and stopped working after one day."}'
```

### Validation behavior

- `text` is required.
- It must include at least one non-whitespace character.
- Maximum request text length is 10,000 characters.
- Invalid input returns HTTP `422` with validation details.

## Run automated API tests

Install the requirements, then run:

```bash
python -m pytest -q
```

## Submission demo

The provided `examples/curl_demo.sh` is a short command-line demo. On Windows, copy the `curl` commands from this README into PowerShell. In the terminal output, capture both `/health` and `/predict` for the screenshot deliverable.

## Publish to GitHub

Create a new repository on GitHub (for example, `sentiment-model-api`) and then run from the project directory:

```bash
git init
git add .
git commit -m "Add Dockerized sentiment inference API"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sentiment-model-api.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository URL shown by GitHub. The repository must include the API code, Dockerfile, README, sample requests, and the model/vectorizer artifacts, as required for this project's inference API to run without retraining.
