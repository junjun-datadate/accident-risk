# app/streamlit_app.py
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="事故リスク優先度リスト", layout="wide")

MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")

st.title("事故リスクのトップ抽出（CSVアップ→スコア→上位x%出力）")

# --- 1) モデル/期待カラムのロード -------------------------------------------
@st.cache_resource
def load_model_and_cols():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("artifacts/model.joblib が見つかりません。まず train.py を実行してください。")
    pipe = joblib.load(MODEL_PATH)

    exp_num, exp_cat = [], []
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        exp_num = m.get("num_features", []) or []
        exp_cat = m.get("cat_features", []) or []
    expected_cols = list(dict.fromkeys([*exp_num, *exp_cat]))  # 順序保持で重複除去
    return pipe, expected_cols

try:
    pipe, expected_cols = load_model_and_cols()
    st.success("モデルを読み込みました。")
except Exception as e:
    st.error(f"モデル読み込みに失敗：{e}")
    st.stop()

# --- 2) CSVアップロード --------------------------------------------------------
st.subheader("① テスト用CSVをアップロード")
up = st.file_uploader("test と同じ列構成のCSVを選んでください（ヘッダーあり, UTF-8）", type=["csv"])

if up is None:
    st.info("右上のメニューからサンプルCSVを用意してもOKです。ファイルを選ぶと自動でスコア計算に進みます。")
    st.stop()

try:
    df = pd.read_csv(up)
except Exception as e:
    st.error(f"CSVの読み込みに失敗しました：{e}")
    st.stop()

st.write("アップロードしたデータ（先頭表示）")
st.dataframe(df.head(20), use_container_width=True)

# 期待する特徴量カラムの整形（不足はNaN追加、余分は自動で無視）
X = df.copy()
# もし期待カラムを保存していない/空なら、アップロードCSVの全列を使う
use_cols = expected_cols if expected_cols else X.columns.tolist()

missing = [c for c in use_cols if c not in X.columns]
for c in missing:
    X[c] = np.nan  # SimpleImputer が処理してくれる

# 不要列は ColumnTransformer が無視するが、順序をそろえておくと安全
X = X[use_cols]

# --- 3) スコア計算 -------------------------------------------------------------
st.subheader("② トップ x% を抽出")
with st.spinner("予測スコアを計算中..."):
    try:
        pred = pipe.predict(X)  # 0〜1のリスク想定（train.pyと同じモデル）
    except Exception as e:
        st.error(f"予測でエラー：{e}\n\n列の不足/型が原因の可能性があります。train時の列と一致しているかご確認ください。")
        st.stop()

df_scored = df.copy()
df_scored["pred_risk"] = np.clip(pred, 0.0, 1.0)

# 任意のID列（出力に残したい列）を選択
keep_cols = st.multiselect(
    "出力に含める列（ID/地点名など任意で）",
    options=list(df_scored.columns),
    default=[c for c in df_scored.columns if c != "pred_risk"][:1],  # デフォルトで先頭1列
    help="スコア列（pred_risk）は自動で含まれます"
)

pct = st.slider("抽出割合（%）", min_value=1, max_value=50, value=10, step=1)
k = max(1, math.ceil(len(df_scored) * (pct / 100)))

# 上位抽出
top_idx = np.argsort(df_scored["pred_risk"].values)[-k:][::-1]
top_df = df_scored.iloc[top_idx].copy()
order_cols = [*keep_cols, "pred_risk"] if keep_cols else ["pred_risk"]
top_df = top_df[order_cols].reset_index(drop=True)

# サマリー
st.markdown(
    f"- 件数: **{len(df_scored)}** 行中、上位 **{pct}% = {k} 行** を抽出\n"
    f"- 予測リスク：平均 **{df_scored['pred_risk'].mean():.3f}** / 上位平均 **{top_df['pred_risk'].mean():.3f}**"
)

st.write("抽出結果（上位から表示・最大100行）")
st.dataframe(top_df.head(100), use_container_width=True)

# --- 4) CSV出力 ---------------------------------------------------------------
st.subheader("③ CSVで出力")
csv_bytes = top_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"CSVを保存（top_{pct}pct.csv）",
    data=csv_bytes,
    file_name=f"top_{pct}pct.csv",
    mime="text/csv",
)
