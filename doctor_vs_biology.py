"""
Model settings are the same as the paper's: depth 4, learning rate 0.2, 40 trees, sepsis rows weighted 40x,
threshold 0.5. Scoring uses the official challenge utility function.

NOTE: SAVE THIS FILE IN THE SAME FOLDER as the "submit-cinc2019" folder (not inside it), like this:
        some_folder/
            submit-cinc2019/
            doctor_vs_biology.py

REQUIREMENTS (download/install before running):
  1. Python 3.9 or newer.
  2. Packages:   pip install numpy pandas scikit-learn "xgboost>=2.0"
  3. The submit-cinc2019 folder from the paper's repo (training_setA and training_setB folders) 
  (the PhysioNet 2019 challenge data, ~40,000 .psv files)
  
  *Tested with Python 3.11, numpy 2.4, pandas 3.1, scikit-learn 1.9, xgboost 3.2.*

To run:
        python doctor_vs_biology.py     (5 seeds, took me <20mins on a laptop)
        python doctor_vs_biology.py --seeds 0 --quick   (10% of patients, ~1 min smoke test)
        python doctor_vs_biology.py --device cuda   (GPU)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "submit-cinc2019")   # this script must sit in the same folder as submit-cinc2019
sys.dont_write_bytecode = True
sys.path.append(PROJECT)
from evaluate_sepsis_score import compute_prediction_utility

PAST_HOURS = 5
NUM_TESTS = 34                         # columns 0-7 vital signs, 8-33 lab tests
LABS = list(range(8, 34))
AGE_GENDER = [34, 35]
ADMIN = [36, 37, 38, 39]               # Unit1, Unit2, HospAdmTime, ICULOS
HOURLY_VITALS = [0, 1, 3, 4, 5, 6]     # HR, O2Sat, SBP, MAP, DBP, Resp (measured around every hour)
SNAPSHOT = [2, 7] + LABS               # Temp, EtCO2, labs (measured irregularly)

PARAMS = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.2,
          "scale_pos_weight": 40, "tree_method": "hist"}
ROUNDS = 40
MODELS = ["1. paper (everything)", "2. doctor behaviour only", "3. biology only", "4. ICU time/admin only"]


# ----------------------------------------------------------------------------------------------------
#DATA


def load_patients(quick):
    values, labels, hospital = [], [], []
    for h, folder in enumerate(["training_setA", "training_setB"]):
        path = os.path.join(PROJECT, folder)
        names = sorted(n for n in os.listdir(path) if n.endswith(".psv"))
        if quick:
            names = names[::10]
        for name in names:
            df = pd.read_csv(os.path.join(path, name), sep="|")
            values.append(df.iloc[:, :40].to_numpy(np.float32))
            labels.append(df["SepsisLabel"].to_numpy(np.int8))
            hospital.append(h)
    return values, labels, np.array(hospital)


def forward_fill(v):
    """Carry the last measured value forward; NaN before the first measurement."""
    idx = np.where(~np.isnan(v), np.arange(len(v))[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    return v[idx, np.arange(v.shape[1])]


def add_past_hours(x):
    """Current hour + previous 5 hours side by side (zeros before ICU admission)."""
    blocks = [x]
    for lag in range(1, PAST_HOURS + 1):
        shifted = np.zeros_like(x)
        shifted[lag:] = x[:-lag]
        blocks.append(shifted)
    return np.hstack(blocks)


# ---------------------------------------------------------------------------
#FEATURES


def paper_features(v, column_means):
    filled = forward_fill(v)
    filled = np.where(np.isnan(filled), column_means, filled)       # average before first reading
    change = np.vstack([np.zeros((1, 40)), np.diff(filled, axis=0)])  # 1-hour changes
    flags = (~np.isnan(v)).astype(np.float32)                        # was-it-measured, all 40
    return np.hstack([filled, change, flags]).astype(np.float32), np.zeros((len(v), 0), np.float32)


def doctor_features(v):
    measured = ~np.isnan(v[:, :NUM_TESTS])
    count = np.cumsum(measured, axis=0)
    shifted24 = np.vstack([np.zeros((24, NUM_TESTS)), count[:-24]]) if len(v) > 24 else np.zeros_like(count)
    count_24h = count - shifted24
    hour = np.arange(len(v))[:, None]
    last = np.maximum.accumulate(np.where(measured, hour, -1), axis=0)
    hours_since = np.where(last >= 0, hour - last, np.nan)
    lab_count = count[:, [i for i in LABS]].sum(1)
    shifted6 = np.r_[np.zeros(6), lab_count[:-6]] if len(v) > 6 else np.zeros_like(lab_count)
    labs_6h = (lab_count - shifted6)[:, None]
    current = np.hstack([count, count_24h, hours_since, labs_6h]).astype(np.float32)
    return measured.astype(np.float32), current


def admin_features(v):
    return np.zeros((len(v), 0), np.float32), v[:, ADMIN].astype(np.float32)


def biology_reference(train_values):
    """Pools of real measured values, and typical hour-to-hour change of each vital (training patients only)."""
    allv = np.vstack(train_values)
    pools = {i: allv[:, i][~np.isnan(allv[:, i])] for i in HOURLY_VITALS + SNAPSHOT}
    step_sd = {}
    for i in HOURLY_VITALS:
        d = np.concatenate([np.diff(v[:, i]) for v in train_values if len(v) > 1])
        d = d[~np.isnan(d)]
        step_sd[i] = float(d.std()) if len(d) else 0.0
    return pools, step_sd


def biology_features(v, pools, step_sd, rng):
    filled = forward_fill(v)
    vitals = filled[:, HOURLY_VITALS].copy()
    for k, i in enumerate(HOURLY_VITALS):
        carried = np.isnan(v[:, i]) & ~np.isnan(vitals[:, k])
        vitals[carried, k] += rng.normal(0, step_sd[i], carried.sum())
        never = np.isnan(vitals[:, k])
        if never.any() and len(pools[i]):
            vitals[never, k] = rng.choice(pools[i]) + rng.normal(0, step_sd[i], never.sum())
    latest = filled[:, SNAPSHOT].copy()
    for k, i in enumerate(SNAPSHOT):
        never = np.isnan(latest[:, k])
        if never.any() and len(pools[i]):   # a variable never recorded in training stays missing
            latest[never, k] = rng.choice(pools[i])
    current = np.hstack([latest, v[:, AGE_GENDER]]).astype(np.float32)
    return vitals.astype(np.float32), current


# ---------------------------------------------------------------------------
#TRAINING / SCORING

def stack(features, ids):
    """Build one big matrix for these patients without making extra copies."""
    first_hist, first_cur = features[ids[0]]
    width = first_hist.shape[1] * (PAST_HOURS + 1) + first_cur.shape[1]
    X = np.empty((sum(len(features[i][0]) for i in ids), width), np.float32)
    r = 0
    for i in ids:
        hist, cur = features[i]
        n = len(hist)
        X[r:r + n] = np.hstack([add_past_hours(hist), cur])
        r += n
    return X


def train(features, labels, ids, seed, args):
    X = stack(features, ids)
    data = xgb.QuantileDMatrix(X, np.concatenate([labels[i] for i in ids]))
    del X
    params = dict(PARAMS, seed=seed, nthread=args.nthread, device=args.device)
    return xgb.train(params, data, num_boost_round=ROUNDS)


def predict(model, features, ids):
    probs = model.inplace_predict(stack(features, ids))
    return np.split(probs, np.cumsum([len(features[i][0]) for i in ids])[:-1])


def normalized_utility(labels, predictions):
    observed = best = inaction = 0
    for y, p in zip(labels, predictions):
        ideal = np.zeros(len(y))
        if y.any():
            onset = np.argmax(y) + 6
            ideal[max(0, onset - 12): onset + 4] = 1
        observed += compute_prediction_utility(y, p, check_errors=False)
        best += compute_prediction_utility(y, ideal, check_errors=False)
        inaction += compute_prediction_utility(y, np.zeros(len(y)), check_errors=False)
    return (observed - inaction) / (best - inaction)


def score(labels, probs):
    utility = normalized_utility(labels, [(p > 0.5).astype(int) for p in probs])
    auroc = roc_auc_score(np.concatenate(labels), np.concatenate(probs))
    return utility, auroc


# ---------------------------------------------------------------------------
#EXPERIMENT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true", help="10%% of patients, for a smoke test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "doctor_vs_biology.csv"))
    args = ap.parse_args()

    t0 = time.time()
    values, labels, hospital = load_patients(args.quick)
    has_sepsis = np.array([y.any() for y in labels])
    print(f"Loaded {len(values)} patients ({time.time() - t0:.0f}s)", flush=True)

    all_ids = np.arange(len(values))
    column_means = np.nanmean(np.vstack(values), axis=0)
    fixed = {MODELS[0]: [paper_features(v, column_means) for v in values],
             MODELS[1]: [doctor_features(v) for v in values],
             MODELS[3]: [admin_features(v) for v in values]}

    def features_for(model_name, train_ids, seed):
        if model_name != MODELS[2]:
            return fixed[model_name]
        pools, step_sd = biology_reference([values[i] for i in train_ids])
        rng = np.random.default_rng(seed)
        return [biology_features(v, pools, step_sd, rng) for v in values]

    rows = []

    def record(**r):
        rows.append(r)
        print(f"  {r['test']:<22} {r['model']:<26} seed={r['seed']} "
              f"utility={r['utility']:.3f} AUROC={r['auroc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Test 1: pooled A+B, 3-fold CV (every patient tested once by a model that never saw them)
    for seed in args.seeds:
        folds = list(StratifiedKFold(3, shuffle=True, random_state=seed)
                     .split(all_ids, has_sepsis * 2 + hospital))
        for name in MODELS:
            probs = [None] * len(values)
            for train_ids, test_ids in folds:
                feats = features_for(name, train_ids, seed)
                model = train(feats, labels, train_ids, seed, args)
                for i, p in zip(test_ids, predict(model, feats, test_ids)):
                    probs[i] = p
            u, a = score(labels, probs)
            record(test="pooled A+B (3-fold)", model=name, seed=seed, utility=u, auroc=a)

    # Test 2: A->B and B->A
    for seed in args.seeds:
        for h, other in ((0, 1), (1, 0)):
            in_h = np.flatnonzero(hospital == h)
            train_ids, held_ids = train_test_split(in_h, test_size=1 / 3, stratify=has_sepsis[in_h],
                                                   random_state=seed)
            other_ids = np.flatnonzero(hospital == other)
            direction = f"{'AB'[h]}->{'AB'[other]}"
            for name in MODELS:
                feats = features_for(name, train_ids, seed)
                model = train(feats, labels, train_ids, seed, args)
                for label, ids in ((f"{direction} same hospital", held_ids),
                                   (f"{direction} new hospital", other_ids)):
                    u, a = score([labels[i] for i in ids], predict(model, feats, ids))
                    record(test=label, model=name, seed=seed, utility=u, auroc=a)

    #Summary
    df = pd.DataFrame(rows)
    summary = df.groupby(["test", "model"])[["utility", "auroc"]].agg(["mean", "std"]).round(3)
    print("\n" + summary.to_string())
    print(f"\nPer-run results saved to {args.out}  (total {(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
