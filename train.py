import os, json, time
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 42
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TARGET = "accident_risk"
ART_DIR = "artifacts"

for p in [TRAIN_PATH, TEST_PATH, SAMPLE_SUB]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} がありません。data/ に3ファイルを置いてください。")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB)

X = train.drop(columns=[TARGET])
y = train[TARGET].astype(float)

num_cols_raw = X.select_dtypes(include=["number"]).columns.tolist()
low_card_ints = [c for c in num_cols_raw
                 if pd.api.types.is_integer_dtype(X[c]) and X[c].nunique(dropna=True) <= 20]
cat_cols = [c for c in X.columns if (c not in num_cols_raw)] + low_card_ints
num_cols = [c for c in num_cols_raw if c not in low_card_ints]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ],
    remainder="drop"
)

model = HistGradientBoostingRegressor(random_state=SEED)
pipe = Pipeline([("prep", preprocess), ("reg", model)])

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
rmse_mean = -scores.mean(); rmse_std = scores.std()
print(f"CV RMSE: mean={rmse_mean:.6f}  std={rmse_std:.6f}")

pipe.fit(X, y)
pred = pipe.predict(test)
sub = sample.copy()
sub.iloc[:, 1] = np.clip(pred, 0.0, 1.0)
sub.to_csv("submission.csv", index=False)

os.makedirs(ART_DIR, exist_ok=True)
joblib.dump(pipe, f"{ART_DIR}/model.joblib")
json.dump({
    "cv_rmse_mean": float(rmse_mean),
    "cv_rmse_std": float(rmse_std),
    "seed": SEED,
    "num_features": num_cols,
    "cat_features": cat_cols,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}, open(f"{ART_DIR}/metrics.json","w"), ensure_ascii=False, indent=2)

print("Saved: submission.csv, artifacts/model.joblib, artifacts/metrics.json")
