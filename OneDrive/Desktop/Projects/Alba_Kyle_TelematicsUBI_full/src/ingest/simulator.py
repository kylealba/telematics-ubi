import argparse, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, time

def simulate_events(n_drivers=200, days=30, seed=42):
    rng = np.random.default_rng(seed)
    drivers = [f"D-{i:03d}" for i in range(1, n_drivers+1)]
    rows = []
    now = int(time.time())
    secs_per_day = 24*3600
    for d in drivers:
        base_risk = rng.beta(2, 5)  # heterogeneous driver profiles
        trips_per_day = rng.integers(1, 5)
        for day in range(days):
            day_start = now - (days - day) * secs_per_day
            for _ in range(trips_per_day):
                segments = 60  # 60 segments per trip
                for _s in range(segments):
                    ts = (day_start + rng.integers(0, secs_per_day)) * 1000
                    speed = max(0, rng.normal(55, 15))
                    long_acc = rng.normal(0, 1.2)
                    lat_acc = rng.normal(0, 0.8)
                    brake = 1 if (rng.random() < (0.03 + 0.2*base_risk) and long_acc < -1.5) else 0
                    urban = 1 if rng.random() < 0.6 else 0
                    night = 1 if rng.random() < 0.25 else 0
                    phone = 1 if rng.random() < (0.01 + 0.1*base_risk) else 0
                    miles = max(0.0, rng.normal(0.3, 0.1))
                    rows.append([d, ts, speed, long_acc, lat_acc, brake, urban, night, phone, miles])

    df = pd.DataFrame(rows, columns=[
        "driver_id","ts","speed_kmh","long_accel","lateral_accel","brake_flag","urban_flag","night_flag","phone_use_flag","miles"
    ])
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_drivers", type=int, default=200)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", type=str, default="data/sample_events.parquet")
    args = ap.parse_args()

    df = simulate_events(args.n_drivers, args.days)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.out)
    print(f"Wrote {args.out} with {len(df):,} rows.")
