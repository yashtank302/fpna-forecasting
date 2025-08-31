import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# ---- Paths ----
DATA_PATH = Path("data/fpna_dataset.xlsx")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---- Load & prep data ----
df = pd.read_excel(DATA_PATH)
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values(["Department", "Month"]).reset_index(drop=True)

# Features: time index + lags (use past info only)
df["t"] = df.groupby("Department").cumcount()
df["Rev_Lag1"] = df.groupby("Department")["Actual_Revenue"].shift(1)
df["Cost_Lag1"] = df.groupby("Department")["Actual_Cost"].shift(1)
df["EBIT_Lag1"] = df.groupby("Department")["EBIT_Actual"].shift(1)

# Drop rows where lags are NaN (first month per department)
dfm = df.dropna(subset=["Rev_Lag1", "Cost_Lag1", "EBIT_Lag1"]).copy()

# One model for all departments -> department dummies
dfm = pd.get_dummies(dfm, columns=["Department"], drop_first=True)

# ---- Time-based train/test split (last 4 months as test) ----
cutoff = dfm["Month"].max() - pd.DateOffset(months=4)
train = dfm[dfm["Month"] <= cutoff].copy()
test  = dfm[dfm["Month"] >  cutoff].copy()

features = [c for c in dfm.columns if c.startswith("Department_")] + ["t", "Rev_Lag1", "Cost_Lag1", "EBIT_Lag1"]
target = "Actual_Revenue"

X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]

# ---- Fit & predict ----
model = LinearRegression().fit(X_train, y_train)
test["Pred_Revenue"] = model.predict(X_test)

# Join back Department for readable output
test_join = test.merge(df[["Month", "Department", "Actual_Revenue"]], on=["Month", "Actual_Revenue"], how="left")
out = test_join[["Month", "Department", "Actual_Revenue", "Pred_Revenue"]].sort_values(["Department", "Month"])

# ---- Metrics ----
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),             # avg money-off
        "MAPE": mean_absolute_percentage_error(y_true, y_pred), # avg % off
        "R2": r2_score(y_true, y_pred)                          # closer to 1 = better
    }

overall = metrics(out["Actual_Revenue"], out["Pred_Revenue"])

# Per-department too
rows = [dict(Department="ALL", **overall)]
for dept, g in out.groupby("Department"):
    m = metrics(g["Actual_Revenue"], g["Pred_Revenue"])
    m["Department"] = dept
    rows.append(m)
metrics_df = pd.DataFrame(rows)

# ---- Save outputs ----
out.to_csv(OUT_DIR / "linear_predictions.csv", index=False)
metrics_df.to_csv(OUT_DIR / "linear_metrics.csv", index=False)

# Plot (Operations)
ops = out[out["Department"] == "Operations"].sort_values("Month")
plt.figure()
plt.plot(ops["Month"], ops["Actual_Revenue"], label="Actual Revenue")
plt.plot(ops["Month"], ops["Pred_Revenue"], label="Predicted Revenue")
plt.title("Operations: Actual vs Predicted — Linear Regression")
plt.xlabel("Month"); plt.ylabel("Revenue"); plt.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "linear_ops_plot.png"); plt.close()

print("Done. Files saved in outputs/:")
print(" - linear_predictions.csv")
print(" - linear_metrics.csv")
print(" - linear_ops_plot.png")

