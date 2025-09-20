# Architecture & Design

## Component View
1. **Data Ingest (POC):**
   - `simulator.py` produces per-event telemetry (timestamped speed, accel, braking flags, GPS tile, phone use).
   - In production, replace with hardware dongle or smartphone SDK; publish to a message bus (e.g., Kafka).
2. **Processing:**
   - `feature_pipeline.py` aggregates events to driver-level features over a rolling window (e.g., 30 days):
     - mean_speed, harsh_brake_rate, night_miles_pct, urban_miles_pct, accel_var, phone_use_pct, total_miles.
3. **Risk Modeling:**
   - Gradient Boosting (POC) predicting `is_high_risk` (synthetic label).
   - Calibrated probabilities → risk score in [0,1].
4. **Pricing Engine (API):**
   - `POST /score` returns `{risk_score, price_factor}`.
   - Simple pricing: `price_factor = base_factor * (1 + alpha * risk_score)` with clamps.
5. **Dashboard:**
   - Streamlit frontend for users to inspect their recent driving metrics and score history.

## Data Schemas

### Event (wide)
| col              | type     | description |
|------------------|----------|-------------|
| driver_id        | str      | pseudo-ID   |
| ts               | int64    | unix ms     |
| speed_kmh        | float    |             |
| long_accel       | float    | m/s^2       |
| lateral_accel    | float    | m/s^2       |
| brake_flag       | int      | 0/1         |
| urban_flag       | int      | 0/1         |
| night_flag       | int      | 0/1         |
| phone_use_flag   | int      | 0/1         |
| miles            | float    | segment mi  |

### Driver Features
| feature              | type   |
|----------------------|--------|
| mean_speed           | float  |
| harsh_brake_rate     | float  |
| night_miles_pct      | float  |
| urban_miles_pct      | float  |
| accel_var            | float  |
| phone_use_pct        | float  |
| total_miles          | float  |

## Security & Privacy
- **Consent & Purpose Limitation**; **Data Minimization**.
- TLS in transit, AES-256 at rest, role-based access (RBAC).
- Rotate tokens/keys; audit logging; retention windows.
- Regionalization & subject rights (CCPA/GDPR).

## Modeling Notes
- Start with GBMs; benchmark vs. logistic regression & random forest.
- Calibrate probabilities (isotonic) for pricing stability.
- Monitor for **fairness** & **proxy bias**; exclude protected attributes.
- Concept drift: schedule periodic retraining & PSA testing.
