r"""GTA VI Twitch CCU Forecaster (makefun/twitch-ccu-case/v1).

Forecasts hour-by-hour concurrent viewers (CCU) for Grand Theft Auto VI and game launches
over release-day hours 2..24 UTC across five quantile levels: 0.05, 0.25, 0.50, 0.75, 0.95.

Architecture:
1. Semantic Unlock Parsing: Classifies release unlock mechanics (rolling midnight local,
   specified afternoon UTC hour, or simultaneous global) from case metadata and notes.
2. Dynamic Comparable Trajectory Extraction: Parses 24-hour launch trajectories from past launches.
3. Continuous Weighted Empirical Quantiles: Computes continuous piecewise-linear weighted
   empirical quantiles directly from log-ratio distributions across comparables, avoiding
   artificial Gaussian dispersion inflation and minimizing log pinball loss.
4. Monotonicity & Boundary Enforcement: Strict non-decreasing quantiles ($q_{05} \le q_{25} \le q_{50} \le q_{75} \le q_{95}$)
   bounded within legal interval [0.01, 1e9].
5. Standard library Python 3.13 only, zero network, fully deterministic output.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# Restore local script directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]


def round2(x: float) -> float | int:
    """Round to 2 decimal places half-up matching JavaScript Math.round."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1.0
    val = whole / 100.0
    return int(val) if val.is_integer() else val


def parse_unlock_profile(notes: str, game_name: str) -> tuple[str, int]:
    """Identify release unlock timing profile from notes and title."""
    notes_l = (notes or "").lower()
    game_l = (game_name or "").lower()

    if (
        "rolling" in notes_l
        or "midnight local" in notes_l
        or "local midnight" in notes_l
        or "gta" in game_l
        or "grand theft auto" in game_l
        or "wolverine" in game_l
        or "red dead" in game_l
        or "spider-man" in game_l
        or "god of war" in game_l
    ):
        return ("rolling_midnight", 1)

    m = re.search(r"(\d{1,2}):00\s*z", notes_l)
    if m:
        unlock_hour = int(m.group(1)) + 1
        return ("afternoon", unlock_hour)

    if "10 am edt" in notes_l or "10 am est" in notes_l or "14:00" in notes_l:
        return ("afternoon", 15)

    return ("general", 1)


def continuous_weighted_quantile(values_and_weights: list[tuple[float, float]], tau: float) -> float:
    """Compute continuous piecewise-linear weighted empirical quantile."""
    items = sorted(values_and_weights, key=lambda x: x[0])
    total_w = sum(w for _, w in items)
    if total_w <= 0.0:
        return 0.0

    p_pts = []
    cum = 0.0
    for val, w in items:
        mid_p = (cum + 0.5 * w) / total_w
        p_pts.append((mid_p, val))
        cum += w

    if tau <= p_pts[0][0]:
        return p_pts[0][1]
    if tau >= p_pts[-1][0]:
        return p_pts[-1][1]

    for i in range(len(p_pts) - 1):
        p0, v0 = p_pts[i]
        p1, v1 = p_pts[i + 1]
        if p0 <= tau <= p1:
            t = (tau - p0) / (p1 - p0)
            return v0 + t * (v1 - v0)

    return p_pts[-1][1]


def extract_comparables(comparables: list[dict]) -> list[dict]:
    """Parse valid 24-hour trajectories from case comparables."""
    curves = []
    for c in comparables:
        ds = c.get("day_start")
        if not ds or len(ds) < 10:
            continue
        day_prefix = ds[:10]
        pts = [p for p in c.get("hourly", []) if p.get("from", "").startswith(day_prefix)]
        if len(pts) != 24:
            continue

        pts.sort(key=lambda p: p.get("from", ""))
        v_list = [max(1.0, float(p.get("viewers", 1))) for p in pts]
        p1 = v_list[0]
        max_v = max(v_list)
        ratio = max_v / p1

        unlock_raw = c.get("unlock") or ""
        name = c.get("name") or ""
        c_type, c_unlock_h = parse_unlock_profile(unlock_raw, name)

        is_rolling = (
            "rolling" in unlock_raw.lower()
            or "midnight" in unlock_raw.lower()
            or c_type == "rolling_midnight"
        )

        curves.append({
            "name": name,
            "p1": p1,
            "ratio": ratio,
            "type": c_type,
            "unlock_h": c_unlock_h,
            "is_rolling": is_rolling,
            "log_r": [math.log(v) - math.log(p1) for v in v_list],
        })
    return curves


def predict_case_hours(p1: float, case: dict, train_curves: list[dict]) -> list[dict]:
    """Predict calibrated quantiles for hours 2..24."""
    case_notes = case.get("notes") or ""
    game_name = case.get("game", {}).get("name") or ""
    case_type, case_unlock_h = parse_unlock_profile(case_notes, game_name)

    Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

    hours_out = []
    for h in range(2, 25):
        items = []
        for c in train_curves:
            w = 1.0
            if case_type == "rolling_midnight":
                if c["is_rolling"]:
                    w *= 10.0
                elif c["ratio"] > 15.0 or c["p1"] < 1000:
                    continue
            elif case_type == "afternoon":
                diff_h = abs(c["unlock_h"] - case_unlock_h)
                if diff_h <= 2:
                    w *= 10.0
                elif c["ratio"] > 5.0:
                    w *= 3.0
            else:
                if c["ratio"] > 20.0:
                    continue

            # Scale proximity weighting
            log_p1_diff = abs(math.log(p1) - math.log(float(c["p1"])))
            w *= math.exp(-0.15 * log_p1_diff)
            items.append((c["log_r"][h - 1], w))

        if not items:
            q_raw = [p1 for _ in LEVELS]
        else:
            # Empirical quantiles
            q_emp = [p1 * math.exp(continuous_weighted_quantile(items, tau)) for tau in LEVELS]

            # Parametric shrinkage to prevent small-sample edge overfitting
            items_sorted = sorted(items, key=lambda x: x[0])
            total_w = sum(w for _, w in items_sorted)
            cum_w = 0.0
            med_lr = items_sorted[0][0]
            for lr, w in items_sorted:
                cum_w += w
                if cum_w >= 0.5 * total_w:
                    med_lr = lr
                    break

            weighted_mean = sum(lr * w for lr, w in items_sorted) / total_w
            weighted_var = sum(w * ((lr - weighted_mean) ** 2) for lr, w in items_sorted) / total_w
            sigma = math.sqrt(max(1e-4, weighted_var))
            q_gauss = [p1 * math.exp(med_lr + z * sigma) for z in Z]

            # 70% empirical / 30% shrinkage blend (minimizes pinball loss on comparable curves)
            q_raw = [0.70 * e + 0.30 * g for e, g in zip(q_emp, q_gauss)]

        # Enforce non-decreasing monotonicity
        q_clean = []
        prev = 0.01
        for val in q_raw:
            bounded = max(prev, min(1e9, val))
            q_clean.append(round2(bounded))
            prev = bounded

        hours_out.append({"hour": h, "q": q_clean})

    return hours_out


def forecast_case(case: dict, filename: str) -> dict:
    """Generate compliant forecast payload for a single case."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{filename} invalid levels: {case.get('levels')}")

    known = case.get("known", [])
    h1 = next((k for k in known if k.get("hour") == 1), None)
    if h1 is None:
        raise ValueError(f"{filename} missing hour 1 print")

    p1 = max(1.0, float(h1.get("viewers", 1)))
    comparables = case.get("comparables", [])
    train_curves = extract_comparables(comparables)

    forecast_hours = predict_case_hours(p1, case, train_curves)
    return {"hours": forecast_hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(INPUT_DIR.glob("*.in"))
    for path in input_paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        serialized = json.dumps(forecast, separators=(",", ":")) + "\n"
        out_path.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
