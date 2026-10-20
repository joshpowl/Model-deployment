"""FastAPI wrapper for the trained PyTorch sentiment classifier."""
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "sentiment_mlp.pth"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"


class SentimentMLP(nn.Module):
    """Architecture used when the supplied model was trained."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_artifacts():
    """Load both artifacts once when the API module starts."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(f"TF-IDF vectorizer not found: {VECTORIZER_PATH}")

    # weights_only=True avoids unrestricted pickle loading for the Torch checkpoint.
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older supported Torch releases.
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = SentimentMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 32)),
        dropout=float(checkpoint.get("dropout", 0.50)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, vectorizer, threshold


model, vectorizer, threshold = load_artifacts()

app = FastAPI(
    title="Sentiment Classification API",
    description="Classify a short review as positive or negative using a saved PyTorch MLP and TF-IDF vectorizer.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Review text to classify (1–10,000 characters).",
        examples=["The product is excellent and works perfectly."],
    )

    @field_validator("text")
    @classmethod
    def text_must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value.strip()


class PredictionResponse(BaseModel):
    text: str
    predicted_sentiment: Literal["positive", "negative"]
    positive_probability: float = Field(ge=0.0, le=1.0)
    negative_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


@app.get("/", tags=["status"])
def root():
    return {
        "message": "Sentiment Classification API is running",
        "health": "/health",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.get("/health", tags=["status"])
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": "cpu"}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_sentiment(payload: PredictionRequest):
    """Return the predicted sentiment and model probabilities for one text."""
    features = vectorizer.transform([payload.text]).toarray().astype(np.float32)
    tensor = torch.from_numpy(features)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    positive_probability = float(probabilities[1])
    negative_probability = float(probabilities[0])
    sentiment = "positive" if positive_probability >= threshold else "negative"
    return PredictionResponse(
        text=payload.text,
        predicted_sentiment=sentiment,
        positive_probability=round(positive_probability, 4),
        negative_probability=round(negative_probability, 4),
        threshold=threshold,
    )
