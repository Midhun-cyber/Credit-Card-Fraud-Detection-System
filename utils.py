import pandas as pd
import joblib
import numpy as np
from io import StringIO

DEFAULT_FEATURES = ['V' + str(i) for i in range(1, 29)] + ['Amount', 'Time']


def robust_read_csv(source):
    if hasattr(source, "read"):
        source.seek(0)
        raw = source.read()
        if not raw:
            raise ValueError("Uploaded file is empty.")
    elif isinstance(source, (bytes, bytearray)):
        raw = source
    else:
        return pd.read_csv(source)

    try:
        text = raw.decode("utf-8", errors="replace")
    except:
        text = str(raw)

    separators = [',', ';', '\t', '|']
    df = None

    for sep in separators:
        try:
            tmp = pd.read_csv(StringIO(text), sep=sep, engine="python")
            if tmp.shape[1] > 1:
                df = tmp
                break
        except:
            pass

    if df is None:
        df = pd.read_csv(StringIO(text), engine="python")

    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    return df


def align_and_fill(df, required_features=DEFAULT_FEATURES):
    df_cols = df.columns.astype(str).tolist()
    aligned = pd.DataFrame(index=df.index)
    for feat in required_features:
        if feat in df_cols:
            aligned[feat] = pd.to_numeric(df[feat], errors='coerce')
        else:
            aligned[feat] = 0.0
    aligned = aligned.fillna(0.0)
    return aligned


def load_artifact(path="../models/model_artifact.joblib"):
    artifact = joblib.load(path)
    pipeline = artifact['pipeline']
    features = artifact['features']
    return pipeline, features


def dynamic_threshold(probs):
    probs = np.asarray(probs)
    if probs.size == 0:
        return 50.0
    p90 = np.percentile(probs, 50)
    mean_plus_std = np.mean(probs) + (1.5*np.std(probs))
    return float(np.clip(min(p90, mean_plus_std), 5, 95))


def verify_pin(pin):
    return pin == "1234"


def process_transactions_batch(model, feature_df, original_df, features):
    aligned_features = align_and_fill(feature_df, required_features=features)
    probs = model.predict_proba(aligned_features)[:, 1] * 100.0
    thresh = dynamic_threshold(probs)
    preds = probs >= thresh

    results_df = pd.DataFrame({
        'ID': original_df.index + 1,
        'Amount': original_df.get('Amount', pd.Series([0] * len(original_df))).apply(
            lambda x: f"{float(x):.2f}" if pd.notnull(x) else "0.00"),
        'Fraud_Probability': probs.round(2),
        'Status': np.where(preds, "🚨 FRAUD", "🟢 SAFE")
    })

    return results_df, probs


def reset_csv_state(st_session_state):
    st_session_state.uploaded_df = None
    st_session_state.csv_processed = False
    st_session_state.csv_results = None
    st_session_state.csv_probs = None
    st_session_state.csv_original = None
    st_session_state.uploaded_filename = None
    st_session_state.premium_unlocked = False
    st_session_state.show_premium_section = False

    try:
        import streamlit as st
        st.cache_data.clear()
    except:
        pass