import warnings
import streamlit as st
import os

os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1000M"
warnings.filterwarnings('ignore')

from utils import (
    robust_read_csv,
    align_and_fill,
    DEFAULT_FEATURES,
    load_artifact,
    dynamic_threshold,
    verify_pin,
    process_transactions_batch,
    reset_csv_state
)

def init_session_state():
    defaults = {
        "csv_processed": False,
        "csv_results": None,
        "csv_probs": None,
        "csv_original": None,
        "premium_unlocked": False,
        "show_premium_section": False,
        "uploaded_filename": None,
        "uploaded_df": None,
        "force_rerun_count": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

model, features = load_artifact("../models/model_artifact.joblib")

def show_basic_premium_csv():
    st.subheader("Basic Premium Analytics")
    col1, col2, col3 = st.columns(3)
    with col1:
        total = len(st.session_state.csv_results)
        st.metric("Total Transactions", total)
    with col2:
        fraud_count = (st.session_state.csv_results['Status'] == "🚨 FRAUD").sum()
        st.metric("Fraud Cases", fraud_count)
    with col3:
        avg_prob = st.session_state.csv_results['Fraud_Probability'].mean()
        st.metric("Avg Fraud Probability", f"{avg_prob:.2f}%")

    st.subheader("Top 10 Highest Risk Transactions")
    top_fraud = st.session_state.csv_results.sort_values(
        'Fraud_Probability', ascending=False
    ).head(10)
    st.dataframe(top_fraud, use_container_width=True)

    if fraud_count > 0:
        fraud_df = st.session_state.csv_results[
            st.session_state.csv_results['Status'] == "🚨 FRAUD"
        ]
        st.subheader("Fraud Amount Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min Fraud Amount", f"${fraud_df['Amount'].min():.2f}")
            st.metric("Avg Fraud Amount", f"${fraud_df['Amount'].mean():.2f}")
        with col2:
            st.metric("Max Fraud Amount", f"${fraud_df['Amount'].max():.2f}")
            st.metric("Total Fraud Amount", f"${fraud_df['Amount'].sum():.2f}")

def show_csv_results():
    if st.session_state.get('uploaded_filename'):
        st.info(f"Dataset: {st.session_state.uploaded_filename}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(st.session_state.csv_results))
    with col2:
        fraud_count = (st.session_state.csv_results['Status'] == "🚨 FRAUD").sum()
        st.metric("Fraud Detected 🚨", fraud_count)
    with col3:
        st.metric("Safe Transactions", len(st.session_state.csv_results) - fraud_count)

    st.subheader("Fraud Cases")
    if fraud_count == 0:
        st.warning("No fraud cases detected.")
    else:
        def highlight(row):
            return ['background-color: rgba(255,0,0,0.12)'] * len(row)

        fraud_only_df = st.session_state.csv_results[
            st.session_state.csv_results['Status'] == "🚨 FRAUD"
        ]
        st.dataframe(
            fraud_only_df.style.apply(highlight, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format="%.2f",
                )
            }
        )

    st.subheader("Download Results")
    fraud_only_df = st.session_state.csv_results[
        st.session_state.csv_results['Status'] == "🚨 FRAUD"
    ]
    st.download_button(
        "Download",
        fraud_only_df.to_csv(index=False),
        file_name="fraud_only_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Premium Analytics Available")
    st.info("Click below to unlock advanced analytics and detailed insights.")
    if st.button("Open Premium Mode", type="secondary", use_container_width=True):
        st.session_state.show_premium_section = True
        st.rerun()

def render_home_page():
    st.title("Credit Card Fraud Detection System")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("About This Project")
        st.markdown("""
        Machine-learning-based credit card fraud detection.

        Features
        - CSV Upload & Batch Processing
        - Smart Auto Threshold
        - Fraud-only Download
        - Premium Analytics
        """)
    with col2:
        st.subheader("Quick Start")
        st.markdown("""
        1. Train Model:
        ```
        python train_pipeline.py --csv ../data/creditcard.csv --out ../models/model_artifact.joblib --model xgb
        ```
        2. Upload CSV for batch processing
        3. Click Open Premium Mode after analysis
        """)

    st.markdown("---")
    if model:
        st.subheader("Model Information")
        c1, c2 = st.columns(2)
        with c1:
            algo = type(model.steps[-1][1]).__name__ if hasattr(model, "steps") else type(model).__name__
            st.metric("Algorithm", algo)
        with c2:
            st.metric("Dataset", "Kaggle")
    else:
        st.warning("Model not loaded.")

def render_csv_upload_page():
    st.title("CSV Upload & Batch Processing")
    st.markdown("---")
    if model is None:
        st.error("Please train the model first!")
        st.stop()

    if not st.session_state.csv_processed:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="File may contain Time, Amount, V1..V28",
            key="csv_uploader"
        )

        if uploaded_file:
            if st.session_state.get('uploaded_df') is None:
                try:
                    with st.spinner("Loading CSV..."):
                        original_df = robust_read_csv(uploaded_file)
                        original_df = original_df.reset_index(drop=True).copy()
                        original_df = original_df.loc[:, ~original_df.columns.str.contains('^Unnamed')]
                        original_df['ID'] = original_df.index + 1
                    st.session_state.uploaded_df = original_df
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"File loaded: {uploaded_file.name}")
                    st.dataframe(original_df.head(), hide_index=True)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

            if st.session_state.get('uploaded_df') is not None:
                if st.button("Process All Transactions", type="primary", key="process_btn"):
                    with st.spinner("Processing transactions..."):
                        original_df = st.session_state.uploaded_df
                        feature_df = original_df.copy()
                        results_df, probs = process_transactions_batch(
                            model, feature_df, original_df, features
                        )
                        st.session_state.csv_original = original_df
                        st.session_state.csv_results = results_df
                        st.session_state.csv_probs = probs
                        st.session_state.csv_processed = True
                        st.rerun()
        return

    if st.session_state.csv_processed and not st.session_state.get('show_premium_section', False):
        show_csv_results()
        return

    if st.session_state.get('premium_unlocked', False) and st.session_state.csv_processed:
        if st.button("Back to Results", type="secondary", use_container_width=True):
            st.session_state.premium_unlocked = False
            st.session_state.show_premium_section = False
            st.rerun()

        st.header("Premium Analytics")
        st.markdown("---")
        try:
            from premium_mode import render_premium_csv_analytics
            render_premium_csv_analytics(
                st,
                st.session_state.csv_original,
                st.session_state.csv_results,
                st.session_state.csv_probs
            )
        except ImportError:
            show_basic_premium_csv()
        except Exception as e:
            st.error(f"Error loading premium: {e}")
            show_basic_premium_csv()
        return

    if st.session_state.get('show_premium_section', False) and not st.session_state.get('premium_unlocked', False):
        st.subheader("Enter Premium PIN to Unlock")
        pin_input = st.text_input(
            "PIN (Default: 1234)",
            type="password",
            max_chars=4,
            key="csv_pin",
            label_visibility="collapsed",
            placeholder="Enter 4-digit PIN"
        )

        col1, col2 = st.columns(2)
        with col1:
            submit = st.button("Unlock", type="primary", use_container_width=True)
        with col2:
            cancel = st.button("Cancel", use_container_width=True)

        if submit and verify_pin(pin_input):
            st.session_state.premium_unlocked = True
            st.session_state.show_premium_section = True
            st.success("Premium Mode Unlocked")
            st.rerun()
        elif submit:
            st.error("Incorrect PIN")

        if cancel:
            st.session_state.show_premium_section = False
            st.session_state.premium_unlocked = False
            st.rerun()
        return

init_session_state()

st.set_page_config(page_title="Fraud Detection", layout="wide")

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["Home", "CSV Upload"])

if page == "Home":
    render_home_page()
elif page == "CSV Upload":
    render_csv_upload_page()


