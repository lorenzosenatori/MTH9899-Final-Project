'''
main.py
│
├── IMPORTS (os, glob, pickle, numpy, pandas, etc.)
│
├── HELPERS (standalone functions)
│   ├── wins orize_xs()
│   ├── zscore_xs()
│   ├── rank_xs()
│   ├── ts_standardize()
│   ├── clip_mad_xs()
│   └── safe_rolling()
│
├── DATA LOADING
│   └── load_raw_data(input_dir) → returns (intra_raw, daily_raw)
│
├── FEATURE FUNCTIONS (we have 1, need 5 more)
│   ├── compute_intraday_features(intra_raw) → returns (intra, boundary)     ✅ done
│   ├── aggregate_intraday(intra, intra_raw) → returns intra_agg             ❌ next
│   ├── compute_daily_features(daily_raw) → returns (daily, daily_feat)      ❌ next
│   ├── compute_extended_features(daily) → returns (daily, e_cols)           ❌ todo
│   ├── compute_cross_sectional_features(intra, daily_raw) → returns cs_feat ❌ todo
│   └── merge_and_normalize(...) → returns df                                ❌ todo
│
├── MODE 1
│   └── run_mode1(input_dir, output_dir, start_date, end_date)               ✅ done
│       calls all feature functions, saves one CSV per date
│
├── MODE 2
│   └── run_mode2(input_dir, output_dir, model_dir, start_date, end_date)    ❌ todo
│       loads feature CSVs + pickle, predicts, saves one CSV per date
│
├── CLI
│   ├── parse_args()                                                          ✅ done
│   └── main()                                                                ✅ don




MTH 9899 — Final Project
main.py

Usage:
    Mode 1: python main.py -m 1 -i <raw_data_dir> -o <feature_dir> -s YYYYMMDD -e YYYYMMDD
    Mode 2: python main.py -m 2 -i <feature_dir> -o <pred_dir> -p <model_dir> -s YYYYMMDD -e YYYYMMDD
'''

import argparse
import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════

def winsorize_xs(s, k=3):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return s.clip(med - k * mad, med + k * mad)

def zscore_xs(s):
    sigma = s.std()
    return (s - s.mean()) / sigma if (sigma > 0 and not np.isnan(sigma)) else s * 0.0

def rank_xs(s):
    return s.rank(pct=True)

def ts_standardize(s, window=20):
    mp = max(5, window // 2)
    rm = s.rolling(window, min_periods=mp).mean()
    rs = s.rolling(window, min_periods=mp).std().replace(0, np.nan)
    return (s - rm) / rs

def clip_mad_xs(s, k=5):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return s.clip(med - k * mad, med + k * mad)

def safe_rolling(s, boundary, window, min_periods, func="mean"):
    s2 = s.copy()
    s2.loc[boundary] = np.nan
    r = s2.rolling(window, min_periods=min_periods)
    return getattr(r, func)() if func != "sum" else r.sum()


#------------------------------------------------------#


def load_raw_data(input_dir):
    """Load intraday and daily CSVs from raw input directory."""
    # ── Intraday ──
    intra_files = sorted(glob.glob(os.path.join(input_dir, "intraday*", "*.csv")))
    if not intra_files:
        raise FileNotFoundError(f"No intraday CSVs in {input_dir}/intraday/")
    intra_raw = pd.concat((pd.read_csv(f) for f in intra_files), ignore_index=True)
    intra_raw["Date"] = pd.to_datetime(intra_raw["Date"], format="%Y%m%d")
    intra_raw["Time_str"] = intra_raw["Time"].astype(str).str.strip()
    intra_raw["Time_td"] = pd.to_timedelta(intra_raw["Time_str"])
    intra_raw["Hour"] = intra_raw["Time_td"].dt.total_seconds() / 3600.0
    intra_raw = intra_raw.sort_values(["Id", "Date", "Time_td"]).reset_index(drop=True)

    # ── Daily ──
    daily_files = sorted(glob.glob(os.path.join(input_dir, "daily*", "*.csv")))
    if not daily_files:
        raise FileNotFoundError(f"No daily CSVs in {input_dir}/daily/")
    daily_raw = pd.concat((pd.read_csv(f) for f in daily_files), ignore_index=True)
    daily_raw.rename(columns={"ID": "Id"}, inplace=True)
    daily_raw["Date"] = pd.to_datetime(daily_raw["Date"], format="%Y%m%d")
    daily_raw = daily_raw.sort_values(["Date", "Id"]).reset_index(drop=True)
    daily_raw["AdjClose"]  = daily_raw["Close"]  * daily_raw["PxAdjFactor"]
    daily_raw["AdjOpen"]   = daily_raw["Open"]   * daily_raw["PxAdjFactor"]
    daily_raw["AdjHigh"]   = daily_raw["High"]   * daily_raw["PxAdjFactor"]
    daily_raw["AdjLow"]    = daily_raw["Low"]    * daily_raw["PxAdjFactor"]
    daily_raw["AdjVolume"] = daily_raw["Volume"] / daily_raw["SharesAdjFactor"]

    return intra_raw, daily_raw


#------------------------------------------------------#


def compute_intraday_features(intra_raw):
    """Compute I1–I11 on intraday data. Returns (intra, boundary)."""
    W = 10
    lam = 0.1

    intra = intra_raw.copy().sort_values(["Id", "Date", "Time_td"]).reset_index(drop=True)
    n_intra = len(intra)

    # ── Diffs & boundaries ──
    boundary = (intra["Id"] != intra["Id"].shift(1)) | (intra["Date"] != intra["Date"].shift(1))

    for src, dst in [("CumReturnResid", "dCumRetResid"),
                     ("CumReturnRaw",   "dCumRetRaw"),
                     ("CumVolume",      "dCumVolume")]:
        intra[dst] = intra[src].diff()
        intra.loc[boundary, dst] = np.nan

    intra["FactorComp"]  = intra["CumReturnRaw"] - intra["CumReturnResid"]
    intra["dFactorComp"] = intra["FactorComp"].diff()
    intra.loc[boundary, "dFactorComp"] = np.nan
    intra["d2CumRetResid"] = intra["dCumRetResid"].diff()
    intra["d2CumVolume"]   = intra["dCumVolume"].diff()
    intra.loc[boundary, "d2CumRetResid"] = np.nan
    intra.loc[boundary, "d2CumVolume"]   = np.nan

    # ── Reference snapshots ──
    gsd = intra.groupby(["Id", "Date"])
    intra = intra.merge(gsd["CumReturnResid"].first().rename("CumRetResid_0945"),
                        on=["Id", "Date"], how="left")
    ref_1000 = intra.loc[intra["Time_td"] == pd.Timedelta(hours=10),
                         ["Id", "Date", "CumReturnResid"]].rename(
                         columns={"CumReturnResid": "CumRetResid_1000"})
    intra = intra.merge(ref_1000, on=["Id", "Date"], how="left")

    # I1: Overnight Residual Return
    intra["I1_OvernightResidReturn"] = intra["CumRetResid_0945"]

    # I2: Overnight-vs-Intraday Disconnect
    intra["I2_OvernightIntradayDisconnect"] = (
        intra["CumRetResid_0945"] - (intra["CumReturnResid"] - intra["CumRetResid_0945"])
    )

    # I3: Anchor-Decayed Residual Displacement
    decay = np.exp(-lam * (intra["Hour"].values - 10.0))
    displacement = intra["CumReturnResid"].values - intra["CumRetResid_1000"].values * decay
    sigma_W = safe_rolling(intra["dCumRetResid"], boundary, W, 3, "std").replace(0, np.nan)
    intra["I3_AnchorDecayedDisplacement"] = displacement / sigma_W.values

    # I4: VWAR
    wr = intra["dCumRetResid"] * intra["dCumVolume"]
    intra["I4_VWAR"] = (
        safe_rolling(wr, boundary, W, 3, "sum") /
        safe_rolling(intra["dCumVolume"], boundary, W, 3, "sum").replace(0, np.nan)
    )

    # I5: Kyle's Lambda
    intra["SignedVol"] = intra["dCumVolume"] * np.sign(intra["dCumRetRaw"])
    xy = intra["dCumRetResid"] * intra["SignedVol"]
    cov = (safe_rolling(xy, boundary, W, 3, "mean") -
           safe_rolling(intra["dCumRetResid"], boundary, W, 3, "mean") *
           safe_rolling(intra["SignedVol"], boundary, W, 3, "mean"))
    var_y = safe_rolling(intra["SignedVol"], boundary, W, 3, "var").replace(0, np.nan)
    intra["I5_KyleLambda"] = cov / var_y

    # I6: Ghost Liquidity
    intra["I6_GhostLiquidity"] = np.log1p(
        intra["dCumRetResid"].abs() / np.log1p(intra["dCumVolume"].clip(lower=0))
    )

    # I7: Abnormal Volume
    intra = intra.sort_values(["Id", "Time_td", "Date"]).reset_index(drop=True)
    avg_vol = intra.groupby(["Id", "Time_td"])["CumVolume"].rolling(20, min_periods=5).mean()
    avg_vol = avg_vol.reset_index(level=[0, 1], drop=True).sort_index().replace(0, np.nan)
    intra["I7_AbnormalVolume"] = np.log(intra["CumVolume"] / avg_vol)
    intra = intra.sort_values(["Id", "Date", "Time_td"]).reset_index(drop=True)
    # Recompute boundary after re-sort
    boundary = (intra["Id"] != intra["Id"].shift(1)) | (intra["Date"] != intra["Date"].shift(1))

    # I8: Shadow Vol Divergence
    k = 10
    d2v_x_d2r = intra["d2CumVolume"] * intra["d2CumRetResid"]
    r_xy8 = safe_rolling(d2v_x_d2r, boundary, k, 4, "mean")
    r_d2v = safe_rolling(intra["d2CumVolume"], boundary, k, 4, "mean")
    r_d2r = safe_rolling(intra["d2CumRetResid"], boundary, k, 4, "mean")
    cov8 = r_xy8 - r_d2v * r_d2r
    s_d2v = safe_rolling(intra["d2CumVolume"], boundary, k, 4, "std")
    s_d2r = safe_rolling(intra["d2CumRetResid"], boundary, k, 4, "std")
    intra["I8_ShadowVolDivergence"] = -(cov8 / (s_d2v * s_d2r).replace(0, np.nan))

    # I9: Max Drawdown
    result_mdd = np.full(n_intra, np.nan)
    for stock_id, grp in intra.groupby("Id"):
        idx = grp.index.values
        vals = grp["CumReturnResid"].values.astype(np.float64)
        m = len(vals)
        if m >= W:
            wins = sliding_window_view(vals, W)
            cm = np.maximum.accumulate(wins, axis=1)
            dd = np.max(cm - wins, axis=1)
            local = np.full(m, np.nan)
            local[W - 1:] = dd
            result_mdd[idx] = local
    intra["I9_MaxDrawdown"] = result_mdd

    # I10: Divergence Velocity
    intra["I10_DivergenceVelocity"] = safe_rolling(intra["dFactorComp"], boundary, W, 3, "std")

    # I11: Permutation Entropy
    pat_window = 12 - 3 + 1
    result_pe = np.full(n_intra, np.nan)
    for stock_id, grp in intra.groupby("Id"):
        vals = grp["dCumRetResid"].values.astype(np.float64)
        n = len(vals)
        if n < 12:
            continue
        v0, v1, v2 = vals[:-2], vals[1:-1], vals[2:]
        pid = ((v0 < v1).astype(np.int8) * 4 +
               (v0 < v2).astype(np.int8) * 2 +
               (v1 < v2).astype(np.int8))
        pv = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(v2)
        np_ = len(pid)
        if np_ < pat_window:
            continue
        ind = np.zeros((np_, 8), dtype=np.float64)
        for p in range(8):
            ind[:, p] = ((pid == p) & pv)
        cs = np.vstack([np.zeros((1, 8)), np.cumsum(ind, axis=0)])
        counts = cs[pat_window:] - cs[:-pat_window]
        total = counts.sum(axis=1, keepdims=True)
        total[total == 0] = np.nan
        probs = counts / total
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.where(probs > 0, np.log2(probs), 0.0)
        entropy = -np.sum(probs * log_p, axis=1)
        offset = pat_window - 1 + 2
        idx = grp.index.values
        end = min(offset + len(entropy), n)
        result_pe[idx[offset:end]] = entropy[:end - offset]
    intra["I11_PermutationEntropy"] = result_pe

    return intra, boundary


def aggregate_intraday(intra, intra_raw):
    """Aggregate intraday features at 15:30 snapshot + A1–A3."""
    FCOLS = ["I1_OvernightResidReturn", "I2_OvernightIntradayDisconnect",
             "I3_AnchorDecayedDisplacement", "I4_VWAR", "I5_KyleLambda",
             "I6_GhostLiquidity", "I7_AbnormalVolume", "I8_ShadowVolDivergence",
             "I9_MaxDrawdown", "I10_DivergenceVelocity", "I11_PermutationEntropy"]

    target_time = pd.Timedelta(hours=15, minutes=30)
    snap_1530 = intra[intra["Time_td"] == target_time].copy().reset_index(drop=True)
    keep = ["Id", "Date"] + [c for c in FCOLS if c in snap_1530.columns]
    intra_agg = snap_1530[keep].copy()

    # A1: Realized range (up to 15:30)
    intra_up_to = intra[intra["Time_td"] <= target_time]
    g = intra_up_to.groupby(["Id", "Date"])["CumReturnRaw"]
    intra_agg = intra_agg.merge(
        (g.max() - g.min()).rename("A1_RealizedRange").reset_index(),
        on=["Id", "Date"], how="left")

    # A2: Morning vs afternoon
    mid = pd.Timedelta(hours=12)
    intra_s = intra_up_to.sort_values(["Id", "Date", "Time_td"])
    morning = intra_s[intra_s["Time_td"] <= mid]
    mf = morning.groupby(["Id", "Date"])["CumReturnResid"].first()
    ml = morning.groupby(["Id", "Date"])["CumReturnResid"].last()
    al = intra_s.groupby(["Id", "Date"])["CumReturnResid"].last()
    intra_agg = intra_agg.merge(
        (ml - mf).rename("A2_MorningReturn").reset_index(),
        on=["Id", "Date"], how="left")
    intra_agg = intra_agg.merge(
        (al - ml).rename("A2_AfternoonReturn").reset_index(),
        on=["Id", "Date"], how="left")
    intra_agg["A2_MorningAfternoonDiff"] = intra_agg["A2_MorningReturn"] - intra_agg["A2_AfternoonReturn"]

    # A3: Volume concentration
    tv = intra_s.groupby(["Id", "Date"])["CumVolume"].last().rename("_tv")
    fh = intra_s[intra_s["Time_td"] <= pd.Timedelta(hours=10, minutes=45)]
    vf = fh.groupby(["Id", "Date"])["CumVolume"].last().rename("_vf")
    lh = intra_s[intra_s["Time_td"] >= pd.Timedelta(hours=15)]
    vl = lh.groupby(["Id", "Date"])["CumVolume"].first().rename("_vl")
    vc = pd.concat([tv, vf, vl], axis=1).reset_index()
    tvn = vc["_tv"].replace(0, np.nan)
    vc["A3_VolConcFirst1h"] = vc["_vf"] / tvn
    vc["A3_VolConcLast1h"] = (vc["_tv"] - vc["_vl"]) / tvn
    intra_agg = intra_agg.merge(
        vc[["Id", "Date", "A3_VolConcFirst1h", "A3_VolConcLast1h"]],
        on=["Id", "Date"], how="left")

    return intra_agg


def compute_daily_features(daily_raw):
    """Compute D1–D4 from daily data. Returns (daily, daily_feat)."""
    daily = daily_raw.copy().sort_values(["Id", "Date"]).reset_index(drop=True)
    gs = daily.groupby("Id")

    daily["LogReturn"] = gs["AdjClose"].transform(lambda s: np.log(s / s.shift(1)))
    daily["SimpleReturn"] = gs["AdjClose"].transform(lambda s: s / s.shift(1) - 1)

    # D1: Trailing Realized Volatility Ratio
    rv = gs["LogReturn"].transform(lambda s: s.rolling(21, min_periods=10).std()) * np.sqrt(252)
    ev = daily["EST_VOL"].replace(0, np.nan)
    daily["D1_VolRatio"] = np.log((rv / ev).clip(lower=1e-6))
    daily["D1_VolRatio"] = gs["D1_VolRatio"].transform(lambda s: ts_standardize(s, 63))

    # D2: 5-Day Return Reversal
    daily["D2_raw"] = gs["AdjClose"].transform(lambda s: s / s.shift(5) - 1)
    daily["D2_Ret5d"] = daily["D2_raw"] / gs["D2_raw"].transform(
        lambda s: s.rolling(63, min_periods=20).std()).replace(0, np.nan)

    # D3: Volume Trend
    mv5  = gs["AdjVolume"].transform(lambda s: s.rolling(5, min_periods=3).median())
    mv63 = gs["AdjVolume"].transform(lambda s: s.rolling(63, min_periods=20).median()).replace(0, np.nan)
    daily["D3_VolTrend"] = np.log(mv5 / mv63)
    daily["D3_VolTrend"] = gs["D3_VolTrend"].transform(lambda s: ts_standardize(s, 63))

    # D4: Range / Overnight Gap
    pc = gs["AdjClose"].shift(1)
    daily["D4_RangeGap"] = np.log1p(
        (daily["AdjHigh"] - daily["AdjLow"]) / ((daily["AdjOpen"] - pc).abs() + 1e-8))
    daily["D4_RangeGap"] = gs["D4_RangeGap"].transform(lambda s: ts_standardize(s, 21))

    # Lag all daily features by 1 day
    d_cols = ["D1_VolRatio", "D2_Ret5d", "D3_VolTrend", "D4_RangeGap"]
    for col in d_cols:
        daily[col] = daily.groupby("Id")[col].shift(1)

    daily_feat = daily[["Id", "Date"] + d_cols + ["FREE_FLOAT_PERCENTAGE", "EST_VOL", "MDV_63"]].copy()

    return daily, daily_feat


def compute_extended_features(daily):
    """Compute E1–E10. Returns (daily, E_COLS)."""
    def gid(col):
        return daily.groupby("Id")[col]

    # E1: Rolling Beta (fixed — market stats at date level)
    mkt_ret = daily.groupby("Date")["LogReturn"].mean().rename("_mkt")
    daily = daily.merge(mkt_ret, on="Date", how="left")
    daily["_sxm"] = daily["LogReturn"] * daily["_mkt"]
    _rxy = gid("_sxm").transform(lambda s: s.rolling(63, min_periods=21).mean())
    _rx = gid("LogReturn").transform(lambda s: s.rolling(63, min_periods=21).mean())
    _mkt_by_date = daily.groupby("Date")["_mkt"].first()
    _mkt_rm = _mkt_by_date.rolling(63, min_periods=21).mean().rename("_mkt_rm")
    _mkt_rv = _mkt_by_date.rolling(63, min_periods=21).var().replace(0, np.nan).rename("_mkt_rv")
    daily = daily.merge(_mkt_rm, on="Date", how="left")
    daily = daily.merge(_mkt_rv, on="Date", how="left")
    daily["E1_Beta63d"] = (_rxy - _rx * daily["_mkt_rm"]) / daily["_mkt_rv"]
    daily.drop(columns=["_mkt_rm", "_mkt_rv"], inplace=True)

    # E2: Beta Deviation
    daily["E2_BetaDeviation"] = daily["E1_Beta63d"] - gid("E1_Beta63d").transform(
        lambda s: s.rolling(252, min_periods=63).mean())
    daily.drop(columns=["_sxm", "_mkt"], inplace=True)

    # E3: Corwin-Schultz Spread (fixed — grouped rolling)
    h, l = np.log(daily["AdjHigh"]), np.log(daily["AdjLow"])
    hl = h - l
    hl_prev = np.log(gid("AdjHigh").shift(1)) - np.log(gid("AdjLow").shift(1))
    h2 = pd.concat([h, np.log(gid("AdjHigh").shift(1))], axis=1).max(axis=1)
    l2 = pd.concat([l, np.log(gid("AdjLow").shift(1))], axis=1).min(axis=1)
    daily["_hl2_sum"] = hl**2 + hl_prev**2
    daily["_h2l2_sq"] = (h2 - l2)**2
    beta_cs = daily.groupby("Id")["_hl2_sum"].transform(
        lambda s: s.rolling(21, min_periods=10).mean())
    gamma_cs = daily.groupby("Id")["_h2l2_sq"].transform(
        lambda s: s.rolling(21, min_periods=10).mean())
    daily.drop(columns=["_hl2_sum", "_h2l2_sq"], inplace=True)
    alpha_cs = ((np.sqrt(2 * beta_cs) - np.sqrt(beta_cs)) /
                (3 - 2 * np.sqrt(2)) -
                np.sqrt(gamma_cs / (3 - 2 * np.sqrt(2)))).clip(lower=0)
    daily["E3_CorwinSchultzSpread"] = np.log1p(
        2 * (np.exp(alpha_cs) - 1) / (1 + np.exp(alpha_cs)))

    # E4/E5: MAX5/MIN5
    daily["E4_MAX5"] = gid("SimpleReturn").transform(lambda s: s.rolling(21, min_periods=10).apply(
        lambda w: np.partition(w, -5)[-5:].mean() if len(w) >= 5 else np.nan, raw=True))
    daily["E5_MIN5"] = gid("SimpleReturn").transform(lambda s: s.rolling(21, min_periods=10).apply(
        lambda w: np.partition(w, 5)[:5].mean() if len(w) >= 5 else np.nan, raw=True))

    # E8: MACD
    ema12 = gid("AdjClose").transform(lambda s: s.ewm(span=12, min_periods=8).mean())
    ema26 = gid("AdjClose").transform(lambda s: s.ewm(span=26, min_periods=18).mean())
    daily["E8_MACD"] = (ema12 - ema26) / daily["AdjClose"].replace(0, np.nan)

    # E9: Bollinger Band Position
    sma20 = gid("AdjClose").transform(lambda s: s.rolling(20, min_periods=10).mean())
    std20 = gid("AdjClose").transform(lambda s: s.rolling(20, min_periods=10).std()).replace(0, np.nan)
    daily["E9_BollingerPos"] = (daily["AdjClose"] - sma20) / (2 * std20)

    # E10: Volume Volatility
    vmean = gid("AdjVolume").transform(lambda s: s.rolling(21, min_periods=10).mean())
    vstd  = gid("AdjVolume").transform(lambda s: s.rolling(21, min_periods=10).std())
    daily["E10_VolVolatility"] = vstd / vmean.replace(0, np.nan)

    # TS normalization
    for col, w in {"E1_Beta63d": 252, "E3_CorwinSchultzSpread": 63,
                    "E4_MAX5": 63, "E5_MIN5": 63, "E10_VolVolatility": 63}.items():
        mp = max(5, w // 2)
        rm = gid(col).transform(lambda s: s.rolling(w, min_periods=mp).mean())
        rs = gid(col).transform(lambda s: s.rolling(w, min_periods=mp).std()).replace(0, np.nan)
        daily[col] = (daily[col] - rm) / rs

    # Lag all extended features
    E_COLS = ["E1_Beta63d", "E2_BetaDeviation", "E3_CorwinSchultzSpread",
              "E4_MAX5", "E5_MIN5", "E8_MACD", "E9_BollingerPos", "E10_VolVolatility"]
    for col in E_COLS:
        daily[col] = daily.groupby("Id")[col].shift(1)

    return daily, E_COLS


def compute_cross_sectional_features(intra, daily_raw):
    """Compute C1–C3. Returns cs_feat DataFrame."""
    intra_1530 = intra[intra["Time_td"] <= pd.Timedelta(hours=15, minutes=30)].copy()

    # C1: Residual Momentum Rank
    gid_c = intra_1530.groupby(["Id", "Date"])["CumReturnResid"]
    rr = (gid_c.last() - gid_c.first()).rename("IntradayResidReturn").reset_index()
    rr["C1_ResidMomRank"] = rr.groupby("Date")["IntradayResidReturn"].rank(pct=True)

    # C2: Vortex Score
    n_snaps = 5
    valid = intra_1530.loc[intra_1530["dCumRetResid"].notna(), ["Id", "Date", "dCumRetResid"]]
    last5 = valid.groupby(["Id", "Date"]).tail(n_snaps)
    pairs = last5.groupby(["Date", "Id"])["dCumRetResid"].apply(lambda x: np.sort(x.values))
    pairs = pairs[pairs.apply(len) >= 3].reset_index()
    pairs.columns = ["Date", "Id", "vals"]

    date_stocks = {}
    for date, grp in pairs.groupby("Date"):
        date_stocks[date] = list(zip(grp["Id"], grp["vals"]))

    cdf_cache = {}
    for kk in range(3, n_snaps + 1):
        cdf_cache[kk] = (np.arange(1, kk + 1) / kk, np.arange(0, kk) / kk)

    result_dates, result_ids, result_ks = [], [], []
    for date, stock_list in date_stocks.items():
        pool = np.sort(np.concatenate([sv for _, sv in stock_list]))
        inv_n = 1.0 / len(pool)
        for sid, inc_sorted in stock_list:
            kk = len(inc_sorted)
            pool_cdf = np.searchsorted(pool, inc_sorted, side='right') * inv_n
            s_cdf, s_cdf_bef = cdf_cache[kk]
            ks = max(np.max(np.abs(s_cdf - pool_cdf)),
                     np.max(np.abs(s_cdf_bef - pool_cdf)))
            result_dates.append(date)
            result_ids.append(sid)
            result_ks.append(ks)

    vdf = pd.DataFrame({"Date": result_dates, "Id": result_ids, "C2_VortexScore": result_ks})

    # C3: Liquidity Score
    eps = 1e-4
    dc = daily_raw[["Date", "Id", "MDV_63", "FREE_FLOAT_PERCENTAGE"]].copy()
    dc["C3_LiqScore"] = np.log(dc["MDV_63"] / (dc["FREE_FLOAT_PERCENTAGE"] / 100 + eps))
    dc = dc.sort_values(["Id", "Date"])
    dc["C3_LiqScore"] = dc.groupby("Id")["C3_LiqScore"].transform(lambda s: ts_standardize(s, 63))

    # Merge
    cs_feat = rr[["Id", "Date", "C1_ResidMomRank"]].copy()
    cs_feat = cs_feat.merge(vdf, on=["Id", "Date"], how="left")
    cs_feat = cs_feat.merge(dc[["Id", "Date", "C3_LiqScore"]], on=["Id", "Date"], how="left")

    return cs_feat


def merge_and_normalize(daily_feat, intra_agg, cs_feat, daily, e_cols):
    """Merge all feature sources, apply XS normalization. Returns final df."""
    
    # ── Core merge ──
    # ── Core merge ──
    df = daily_feat.merge(intra_agg, on=["Id", "Date"], how="left")
    df = df.merge(cs_feat, on=["Id", "Date"], how="left")

    # ── Log-transform right-skewed non-negative features ──
    for col in ["I9_MaxDrawdown", "I10_DivergenceVelocity", "A1_RealizedRange"]:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ── TS normalization for intraday magnitude-dependent features ──
    ts_norm_cols = ["I4_VWAR", "I5_KyleLambda", "I6_GhostLiquidity",
                    "I9_MaxDrawdown", "I10_DivergenceVelocity", "A1_RealizedRange"]
    for col in ts_norm_cols:
        if col in df.columns:
            df[col] = df.groupby("Id")[col].transform(lambda s: ts_standardize(s, 20))

    # ── XS normalization for core features ──

    # ── XS normalization for core features ──
    CORE_FEAT = [
        "I1_OvernightResidReturn", "I2_OvernightIntradayDisconnect",
        "I3_AnchorDecayedDisplacement", "I4_VWAR", "I5_KyleLambda",
        "I6_GhostLiquidity", "I7_AbnormalVolume", "I8_ShadowVolDivergence",
        "I9_MaxDrawdown", "I10_DivergenceVelocity", "I11_PermutationEntropy",
        "D1_VolRatio", "D2_Ret5d", "D3_VolTrend", "D4_RangeGap",
        "C1_ResidMomRank", "C2_VortexScore", "C3_LiqScore",
        "A1_RealizedRange", "A2_MorningAfternoonDiff", "A2_MorningReturn",
        "A2_AfternoonReturn", "A3_VolConcFirst1h", "A3_VolConcLast1h",
    ]
    rank_set = {"I5_KyleLambda", "I8_ShadowVolDivergence", "C1_ResidMomRank"}

    for col in CORE_FEAT:
        if col not in df.columns:
            continue
        df[col] = df.groupby("Date")[col].transform(winsorize_xs)
        df[col] = df.groupby("Date")[col].transform(rank_xs if col in rank_set else zscore_xs)

    # ── Merge extended features + XS normalize ──
    E_COLS = [c for c in e_cols if c in daily.columns]
    df = df.merge(daily[["Id", "Date"] + E_COLS], on=["Id", "Date"], how="left")
    for col in E_COLS:
        df[col] = df.groupby("Date")[col].transform(winsorize_xs)
        df[col] = df.groupby("Date")[col].transform(zscore_xs)

    return df


#------------------------------------------------------#



def parse_args():
    parser = argparse.ArgumentParser(description="Feature creation and prediction")
    parser.add_argument("-i", required=True, help="Input directory")
    parser.add_argument("-o", required=True, help="Output directory")
    parser.add_argument("-p", default=None, help="Model directory (Mode 2 only)")
    parser.add_argument("-s", required=True, help="Start date YYYYMMDD")
    parser.add_argument("-e", required=True, help="End date YYYYMMDD")
    parser.add_argument("-m", required=True, type=int, choices=[1, 2], help="Mode: 1=features, 2=predict")
    return parser.parse_args()


#------------------------------------------------------#

def main():
    args = parse_args()
    start_date = pd.Timestamp(args.s)
    end_date = pd.Timestamp(args.e)
    os.makedirs(args.o, exist_ok=True)

    if args.m == 1:
        run_mode1(args.i, args.o, start_date, end_date)
    elif args.m == 2:
        if args.p is None:
            raise ValueError("Mode 2 requires -p (model directory)")
        run_mode2(args.i, args.o, args.p, start_date, end_date)


#------------------------------------------------------#

def run_mode1(input_dir, output_dir, start_date, end_date):
    """Mode 1: Create features from raw data, save one CSV per date."""
    
    # ── Load raw data ──
    print("Loading raw data...")
    intra_raw, daily_raw = load_raw_data(input_dir)
    
    # ── Intraday features ──
    print("Computing intraday features...")
    intra, boundary = compute_intraday_features(intra_raw)
    
    # ── Aggregate at 15:30 ──
    print("Aggregating at 15:30...")
    intra_agg = aggregate_intraday(intra, intra_raw)
    
    # ── Daily features ──
    print("Computing daily features...")
    daily, daily_feat = compute_daily_features(daily_raw)
    
    # ── Extended features ──
    print("Computing extended features...")
    daily, e_cols = compute_extended_features(daily)
    
    # ── Cross-sectional features ──
    print("Computing cross-sectional features...")
    cs_feat = compute_cross_sectional_features(intra, daily_raw)
    
    # ── Merge all ──
    print("Merging features...")
    df = merge_and_normalize(daily_feat, intra_agg, cs_feat, daily, e_cols)
    
    # ── Save per date ──
    dates = df["Date"].unique()
    dates = dates[(dates >= start_date) & (dates <= end_date)]
    dates = np.sort(dates)
    
    print(f"Saving features for {len(dates)} dates...")
    for dt in dates:
        day_df = df[df["Date"] == dt].copy()
        date_str = pd.Timestamp(dt).strftime("%Y%m%d")
        out_path = os.path.join(output_dir, f"{date_str}.csv")
        day_df.to_csv(out_path, index=False)
    
    print(f"Mode 1 complete — {len(dates)} feature files saved to {output_dir}")


def run_mode2(input_dir, output_dir, model_dir, start_date, end_date):
    """Mode 2: Load features + model, make predictions, save one CSV per date."""
    
    # ── Load pickle ──
    pkl_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No model.pkl found in {model_dir}")
    
    with open(pkl_path, "rb") as f:
        model_bundle = pickle.load(f)
    
    models = model_bundle["models"]           # dict: {"ridge": model, "rf": model, "xgb": model}
    scaler = model_bundle["scaler"]           # StandardScaler (fitted on training data)
    feat_cols = model_bundle["feat_cols"]     # list of feature column names
    weights = model_bundle["weights"]         # dict: {"ridge": w, "rf": w, "xgb": w}
    
    # ── Find feature files in date range ──
    feature_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    
    n_saved = 0
    for fpath in feature_files:
        # Parse date from filename (YYYYMMDD.csv)
        basename = os.path.splitext(os.path.basename(fpath))[0]
        try:
            file_date = pd.Timestamp(basename)
        except:
            continue
        
        if file_date < start_date or file_date > end_date:
            continue
        
        # ── Load features ──
        day_df = pd.read_csv(fpath)
        day_df["Date"] = pd.to_datetime(day_df["Date"])
        
        if len(day_df) == 0:
            continue
        
        # ── Prepare feature matrix ──
        # Fill missing feature columns with 0
        for col in feat_cols:
            if col not in day_df.columns:
                day_df[col] = 0.0
        
        X = day_df[feat_cols].fillna(0).values
        
        # ── Predict with each model ──
        preds = {}
        for name, model in models.items():
            if name == "ridge":
                X_scaled = scaler.transform(X)
                preds[name] = model.predict(X_scaled)
            else:
                preds[name] = model.predict(X)
        
        # ── Ensemble (weighted average in normalized space) ──
        pred_norm = np.zeros(len(day_df))
        for name in models:
            pred_norm += weights[name] * preds[name]

        # After computing pred_norm, before scaling back to return space
        pred_norm = clip_mad_xs(pd.Series(pred_norm)).values
        
        # ── Scale back to return space ──
        est_vol = day_df["EST_VOL"].replace(0, np.nan).values
        pred_return = pred_norm * est_vol
        
        # ── Save: Date, Time, Id, Pred ──
        out_df = pd.DataFrame({
            "Date": day_df["Date"].dt.strftime("%Y%m%d"),
            "Time": "15:30:00",
            "Id": day_df["Id"],
            "Pred": pred_return,
        })
        
        date_str = file_date.strftime("%Y%m%d")
        out_path = os.path.join(output_dir, f"{date_str}.csv")
        out_df.to_csv(out_path, index=False)
        n_saved += 1
    
    print(f"Mode 2 complete — {n_saved} prediction files saved to {output_dir}")


#------------------------------------------------------#


if __name__ == "__main__":
    main()