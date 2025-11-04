import streamlit as st, pandas as pd, numpy as np, joblib, json
st.set_page_config(page_title="事故リスク予測デモ", layout="centered")
st.title("事故リスク予測デモ")

pipe = joblib.load("artifacts/model.joblib")
meta = json.load(open("artifacts/metrics.json"))
cols = (meta.get("num_features", []) or []) + (meta.get("cat_features", []) or [])

th = st.slider("危険扱いの基準（しきい値）", 0.0, 1.0, 0.50, 0.01)
up = st.file_uploader("特徴量付きCSV（train/testと同じ列）", type=["csv"])
if up:
    df = pd.read_csv(up)
    X = df[cols] if cols and set(cols).issubset(df.columns) else df.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    pred = pipe.predict(X)
    out = df.copy()
    out["risk"] = np.clip(pred, 0, 1)
    out["alert"] = (out["risk"] >= th).astype(int)
    st.write(out.head())
    st.download_button("結果CSVをダウンロード",
        out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")
else:
    st.info("CSVをアップすると危険度を表示します。")
