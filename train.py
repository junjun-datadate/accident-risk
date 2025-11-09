# train.py（自動で README を更新する版）
import os, json, time, io
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
README = "README.md"

for p in (TRAIN_PATH, TEST_PATH, SAMPLE_SUB):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} がありません。data/ に3ファイルを置いてください。")

# 1) データ
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB)

X = train.drop(columns=[TARGET])
y = train[TARGET].astype(float)

# 2) 列タイプ（低カーディナリティ整数→カテゴリ）
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

# 3) CV（RMSE）
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
rmse_mean = float(-scores.mean())
rmse_std  = float(scores.std())
print(f"CV RMSE: mean={rmse_mean:.6f}  std={rmse_std:.6f}")

# 4) “運用価値”の指標：上位α%の平均とUplift（5/10/20%）
def capture_at_top_alpha(pipe, X, y, alpha=0.10, seed=SEED):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    caps, uplifts = [], []
    for tr, va in kf.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        k = int(np.ceil(len(va) * alpha))
        top_idx = np.argsort(pred)[-k:]
        y_va = y.iloc[va]
        cap  = float(y_va.iloc[top_idx].mean())
        base = float(y_va.mean())
        caps.append(cap)
        uplifts.append(cap / base if base > 0 else np.nan)
    return float(np.mean(caps)), float(np.mean(uplifts))

(cap5, up5)   = capture_at_top_alpha(pipe, X, y, alpha=0.05)
(cap10, up10) = capture_at_top_alpha(pipe, X, y, alpha=0.10)
(cap20, up20) = capture_at_top_alpha(pipe, X, y, alpha=0.20)

# 5) 素朴法との比較（平均だけで予測）
y_mean = float(y.mean())
rmse_naive = float(np.sqrt(np.mean((y - y_mean)**2)))
improve_pct = float((rmse_naive - rmse_mean) / rmse_naive * 100)

# 6) 学習→提出ファイル
pipe.fit(X, y)
pred = pipe.predict(test)
sub = sample.copy()
sub.iloc[:, 1] = np.clip(pred, 0.0, 1.0)
sub.to_csv("submission.csv", index=False)

# 7) 保存
os.makedirs(ART_DIR, exist_ok=True)
joblib.dump(pipe, f"{ART_DIR}/model.joblib")
metrics = {
    "cv_rmse_mean": rmse_mean,
    "cv_rmse_std": rmse_std,
    "rmse_naive": rmse_naive,
    "improve_pct": improve_pct,
    "cap_mean_5": cap5,   "uplift_5": up5,
    "cap_mean_10": cap10, "uplift_10": up10,
    "cap_mean_20": cap20, "uplift_20": up20,
    "seed": SEED,
    "num_features": num_cols,
    "cat_features": cat_cols,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
json.dump(metrics, open(f"{ART_DIR}/metrics.json","w"), ensure_ascii=False, indent=2)
print("Saved: submission.csv, artifacts/model.joblib, artifacts/metrics.json")

# 8) README を自動更新（マーカー内だけ上書き）
def replace_between_markers(path, start_tag, end_tag, new_text):
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
    except FileNotFoundError:
        s = ""
    if start_tag in s and end_tag in s:
        pre = s.split(start_tag)[0]
        post = s.split(end_tag)[1]
        s_new = pre + start_tag + "\n" + new_text.rstrip() + "\n" + end_tag + post
    else:
        # マーカーが無ければ末尾に追記
        block = f"{start_tag}\n{new_text.rstrip()}\n{end_tag}\n"
        s_new = (s + "\n\n" + block) if s else block
    with open(path, "w", encoding="utf-8") as f:
        f.write(s_new)

metrics_lines = [
    f"- 精度：平均だけの方法に比べて誤差を **{improve_pct:.0f}%** 小さく（RMSE {rmse_naive:.4f} → **{rmse_mean:.4f}**）",
    f"- 運用：上位 **5/10/20%** に絞ると平均リスクは **{up5:.2f}x / {up10:.2f}x / {up20:.2f}x**",
]
replace_between_markers(
    README, 
    "<!-- METRICS_START -->",
    "<!-- METRICS_END -->",
    "\n".join(metrics_lines)
)

print("README.md を最新の指標で更新しました。")
