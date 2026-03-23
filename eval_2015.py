# eval_2015.py — run separately from command line
import os, glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ── Load predictions ──
pred_files = sorted(glob.glob("predictions_2015/*.csv"))
preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
preds["Date"] = pd.to_datetime(preds["Date"], format="%Y%m%d")
print(f"Predictions: {len(preds):,} rows, {preds['Date'].nunique()} dates")

# ── Load intraday data (2015 only, for target construction) ──
intra_files = sorted(glob.glob("oos_data/intraday*/*.csv"))
# Only load 2015 files to save memory
intra_2015 = []
for f in intra_files:
    basename = os.path.basename(f).replace(".csv", "")
    try:
        if int(basename[:4]) >= 2014:  # need 2014 end + 2015
            intra_2015.append(pd.read_csv(f))
    except:
        intra_2015.append(pd.read_csv(f))
intra = pd.concat(intra_2015, ignore_index=True)
intra["Date"] = pd.to_datetime(intra["Date"], format="%Y%m%d")
intra["Time_td"] = pd.to_timedelta(intra["Time"].astype(str).str.strip())
print(f"Intraday loaded: {len(intra):,} rows")

# ── Load daily data (for MDV_63 and EST_VOL) ──
daily_files = sorted(glob.glob("oos_data/daily*/*.csv"))
daily = pd.concat([pd.read_csv(f) for f in daily_files], ignore_index=True)
daily.rename(columns={"ID": "Id"}, inplace=True)
daily["Date"] = pd.to_datetime(daily["Date"], format="%Y%m%d")
# Keep only 2015
daily_2015 = daily[daily["Date"].dt.year == 2015][["Date", "Id", "MDV_63", "EST_VOL"]].copy()

# ── Construct target: 24h forward residual return from 15:30 ──
t_1530 = pd.Timedelta(hours=15, minutes=30)
t_1600 = pd.Timedelta(hours=16)

snap_1530 = intra[intra["Time_td"] == t_1530][["Id", "Date", "CumReturnResid"]].rename(
    columns={"CumReturnResid": "resid_1530"})
snap_1600 = intra[intra["Time_td"] == t_1600][["Id", "Date", "CumReturnResid"]].rename(
    columns={"CumReturnResid": "resid_1600"})

target = snap_1530.merge(snap_1600, on=["Id", "Date"], how="inner")
target["ret_1530_to_1600"] = target["resid_1600"] - target["resid_1530"]
target = target.sort_values(["Id", "Date"]).reset_index(drop=True)
target["resid_1530_next"] = target.groupby("Id")["resid_1530"].shift(-1)
target["Date_next"] = target.groupby("Id")["Date"].shift(-1)
target["Target_Resid_24h"] = target["ret_1530_to_1600"] + target["resid_1530_next"]
target["days_gap"] = (target["Date_next"] - target["Date"]).dt.days
target.loc[target["days_gap"] > 4, "Target_Resid_24h"] = np.nan

# Keep only 2015
target_2015 = target[target["Date"].dt.year == 2015][["Id", "Date", "Target_Resid_24h"]].copy()

# ── Merge predictions with actuals ──
merged = preds.merge(target_2015, on=["Id", "Date"], how="inner")
merged = merged.merge(daily_2015, on=["Id", "Date"], how="left")
merged = merged.dropna(subset=["Pred", "Target_Resid_24h", "MDV_63"])

print(f"Matched rows: {len(merged):,}")

# ── Clip actuals at 5 MAD cross-sectionally ──
def clip_mad_xs(s, k=5):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return s.clip(med - k * mad, med + k * mad)

merged["Target_Clipped"] = merged.groupby("Date")["Target_Resid_24h"].transform(clip_mad_xs)

# ── Compute R² ──
w_mdv = np.sqrt(merged["MDV_63"].clip(0).values)
w_vol = 1.0 / merged["EST_VOL"].replace(0, np.nan).values

r2_mdv = r2_score(merged["Target_Clipped"], merged["Pred"], sample_weight=w_mdv)

vol_mask = np.isfinite(w_vol)
r2_vol = r2_score(merged.loc[vol_mask, "Target_Clipped"], 
                   merged.loc[vol_mask, "Pred"], 
                   sample_weight=w_vol[vol_mask])

print(f"\n{'='*50}")
print(f"2015 HOLDOUT RESULTS")
print(f"{'='*50}")
print(f"  OOS R² (√MDV_63):   {r2_mdv:.6f}")
print(f"  OOS R² (1/EST_VOL): {r2_vol:.6f}")
print(f"  Dates:               {merged['Date'].nunique()}")
print(f"  Stocks:              {merged['Id'].nunique()}")
print(f"  Observations:        {len(merged):,}")
print(f"{'='*50}")