from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, os

MODEL_PATH = os.environ.get("MODEL_PATH", "models/risk_model.pkl")
BUNDLE = joblib.load(MODEL_PATH)
MODEL = BUNDLE["model"]
FEATS = BUNDLE["features"]

class ScoreRequest(BaseModel):
    driver_id: str
    mean_speed: float
    harsh_brake_rate: float = Field(ge=0, le=1)
    night_miles_pct: float = Field(ge=0, le=1)
    urban_miles_pct: float = Field(ge=0, le=1)
    weekend_miles_pct: float = Field(ge=0, le=1)
    accel_var: float
    phone_use_pct: float = Field(ge=0, le=1)
    total_miles: float | None = Field(default=0, ge=0)
    rush_hour_miles_pct: float | None = Field(default=None, ge=0, le=1)

class ScoreResponse(BaseModel):
    driver_id: str
    risk_score: float
    price_factor: float

app = FastAPI(title="Telematics Pricing API", version="0.1.0")

def price_factor_from_score(p: float) -> float:
    """Map risk probability to price factor (0.85x–1.25x)."""
    price = 1.0 + 0.8 * (p - 0.5)
    return float(min(max(price, 0.85), 1.25))


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    x = [[getattr(req, f) for f in FEATS]]
    s = float(MODEL.predict_proba(x)[0][1])
    pf = price_factor_from_score(s)
    return ScoreResponse(driver_id=req.driver_id, risk_score=s, price_factor=pf)

@app.get("/health")
def health():
    return {"ok": True}
