import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="XAI-AD · Healthcare Anomaly Detection",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0f1729; color: #ffffff; }
    .stApp { background-color: #0f1729; }
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label { color: #ffffff !important; }
    h1, h2, h3 { color: #38bdf8 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem !important; font-weight: 600 !important; }
    [data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-size: 14px !important; }
    [data-testid="stMetricDelta"] { color: #4ade80 !important; }
    section[data-testid="stSidebar"] { background-color: #1a2540 !important; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] a { color: #38bdf8 !important; }
    [data-testid="stSidebarContent"] { background-color: #1a2540 !important; }
    .st-emotion-cache-1cypcdb, .st-emotion-cache-dvne4q { background-color: #1a2540 !important; }
    .stButton > button { background-color: #2563eb !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    .stButton > button:hover { background-color: #1d4ed8 !important; }
    .stRadio label { color: #ffffff !important; }
    .stDataFrame { background: #1e2d4a !important; }
    [data-testid="stDataFrame"] * { color: #ffffff !important; }
    .stSelectbox label, .stSelectbox div { color: #ffffff !important; }
    .stSlider label { color: #ffffff !important; }
    .stCaption { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    contamination = st.slider("Contamination (expected anomaly %)", 0.01, 0.30, 0.10, 0.01)
    n_estimators = st.slider("Number of trees", 50, 300, 100, 10)
    max_rows = st.slider("Max rows to analyse", 1000, 10000, 5000, 500)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built by **Punitha Karmegam**")
    st.markdown("Master Data Intelligence · ISEP Paris")
    st.markdown("[GitHub](https://github.com/punithakarmegam) · [LinkedIn](https://www.linkedin.com/in/punitha-k-6b8393100/)")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🏥 XAI-AD: Explainable Anomaly Detection in Healthcare")
st.markdown("**Isolation Forest + SHAP** — Detect and explain anomalies in patient data")
st.markdown("---")

# ─── FEATURE COLUMNS ─────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "Age", "Gender", "Blood Type", "Medical Condition",
    "Billing Amount", "Room Number", "Admission Type",
    "Medication", "Test Results", "Insurance Provider"
]
DISPLAY_COLS = [
    "Name", "Age", "Gender", "Blood Type", "Medical Condition",
    "Admission Type", "Billing Amount", "Test Results", "Status", "Anomaly_Score"
]

# ─── PREPROCESSING ───────────────────────────────────────────────────────────
def preprocess(df):
    df_enc = df[MODEL_FEATURES].copy()
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc

# ─── DATA LOADING ────────────────────────────────────────────────────────────
st.subheader("📂 Load Data")
data_option = st.radio("Choose data source:", ["Upload your own CSV", "Use sample data"], horizontal=True)

df = None

if data_option == "Upload your own CSV":
    uploaded = st.file_uploader("Upload healthcare_dataset.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        st.success(f"✅ Loaded **{len(df):,} patients** × {len(df.columns)} columns")
    else:
        st.info("👆 Upload your **healthcare_dataset.csv** file")
        st.stop()
else:
    np.random.seed(42)
    n = 1000
    billing = np.concatenate([np.random.normal(25000, 6000, n - 50), np.random.uniform(80000, 150000, 50)])
    df = pd.DataFrame({
        "Name": [f"Patient {i}" for i in range(n)],
        "Age": np.random.randint(18, 90, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Blood Type": np.random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], n),
        "Medical Condition": np.random.choice(["Cancer", "Diabetes", "Obesity", "Asthma", "Hypertension", "Arthritis"], n),
        "Date of Admission": "2024-01-01",
        "Doctor": "Dr. Sample",
        "Hospital": "Sample Hospital",
        "Insurance Provider": np.random.choice(["Aetna", "Medicare", "Blue Cross", "Cigna"], n),
        "Billing Amount": billing,
        "Room Number": np.random.randint(100, 500, n),
        "Admission Type": np.random.choice(["Elective", "Emergency", "Urgent"], n),
        "Discharge Date": "2024-01-10",
        "Medication": np.random.choice(["Aspirin", "Ibuprofen", "Paracetamol", "Penicillin"], n),
        "Test Results": np.random.choice(["Normal", "Abnormal", "Inconclusive"], n),
    })
    st.success(f"✅ Sample dataset loaded — {n:,} patients")

st.dataframe(df.head(8), use_container_width=True)

# ─── RUN MODEL ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Run Anomaly Detection")

if len(df) > max_rows:
    st.warning(f"⚠️ Dataset has {len(df):,} rows. Analysing a sample of {max_rows:,} rows for speed.")
    df_model = df.sample(max_rows, random_state=42).reset_index(drop=True)
else:
    df_model = df.copy().reset_index(drop=True)

if st.button("🚀 Detect Anomalies", type="primary"):
    with st.spinner("Training Isolation Forest and computing SHAP values…"):
        import time

        missing = [c for c in MODEL_FEATURES if c not in df_model.columns]
        if missing:
            st.error(f"❌ Missing columns in your CSV: {missing}")
            st.stop()

        X = preprocess(df_model)
        feature_cols = X.columns.tolist()

        t0 = time.time()
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        model.fit(X)
        train_time = time.time() - t0

        t1 = time.time()
        preds = model.predict(X)
        pred_time = time.time() - t1

        scores = model.decision_function(X)

        df_model["Anomaly"] = preds
        df_model["Anomaly_Score"] = scores.round(4)
        df_model["Status"] = df_model["Anomaly"].map({-1: "🔴 Anomaly", 1: "🟢 Normal"})
        df_model["Billing Amount"] = df_model["Billing Amount"].round(2)

        n_anomalies = (preds == -1).sum()
        n_normal = (preds == 1).sum()
        energy = train_time * 50

        # ── METRICS ──────────────────────────────────────────────────────────
        st.markdown("### 📊 Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Patients", f"{len(df_model):,}")
        c2.metric("🔴 Anomalies", f"{n_anomalies:,}", f"{n_anomalies/len(df_model)*100:.1f}%")
        c3.metric("🟢 Normal", f"{n_normal:,}")
        c4.metric("⏱ Train Time", f"{train_time:.4f}s")
        c5.metric("⚡ Energy", f"{energy:.2f}J")

        # ── ANOMALY TABLE ────────────────────────────────────────────────────
        st.markdown("### 🔴 Detected Anomalies")
        show_cols = [c for c in DISPLAY_COLS if c in df_model.columns]
        anomaly_df = df_model[df_model["Anomaly"] == -1][show_cols].sort_values("Anomaly_Score").reset_index(drop=True)
        st.dataframe(
            anomaly_df.style
            .format({"Billing Amount": "{:,.2f}", "Anomaly_Score": "{:.4f}"})
            .background_gradient(subset=["Billing Amount"], cmap="Reds"),
            use_container_width=True
        )

        # ── CONDITION BREAKDOWN ──────────────────────────────────────────────
        st.markdown("### 🏥 Anomalies by Medical Condition")
        cond_counts = df_model[df_model["Anomaly"] == -1]["Medical Condition"].value_counts()
        fig0, ax0 = plt.subplots(figsize=(10, 3))
        fig0.patch.set_facecolor("#0f1729")
        ax0.set_facecolor("#1e2d4a")
        bars = ax0.bar(cond_counts.index, cond_counts.values, color="#f87171", alpha=0.85)
        ax0.set_ylabel("Count", color="#94a3b8")
        ax0.tick_params(colors="#94a3b8")
        ax0.spines[["top", "right"]].set_visible(False)
        ax0.spines[["left", "bottom"]].set_color("#334155")
        for bar, val in zip(bars, cond_counts.values):
            ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val), ha="center", color="#f8fafc", fontsize=10)
        st.pyplot(fig0)
        plt.close()

        # ── BILLING DISTRIBUTION ─────────────────────────────────────────────
        st.markdown("### 📈 Billing Amount Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0f1729")
        ax.set_facecolor("#1e2d4a")
        ax.hist(df_model[df_model["Anomaly"] == 1]["Billing Amount"], bins=50, color="#38bdf8", alpha=0.7, label="Normal")
        ax.hist(df_model[df_model["Anomaly"] == -1]["Billing Amount"], bins=30, color="#f87171", alpha=0.85, label="Anomaly")
        ax.set_xlabel("Billing Amount (€)", color="#94a3b8")
        ax.set_ylabel("Count", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#1e2d4a", labelcolor="#f8fafc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#334155")
        st.pyplot(fig)
        plt.close()

        # ── ADMISSION TYPE BREAKDOWN ─────────────────────────────────────────
        st.markdown("### 🚨 Anomalies by Admission Type")
        adm_counts = df_model[df_model["Anomaly"] == -1]["Admission Type"].value_counts()
        colors_adm = {"Emergency": "#f87171", "Urgent": "#fb923c", "Elective": "#38bdf8"}
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        fig1.patch.set_facecolor("#0f1729")
        ax1.set_facecolor("#1e2d4a")
        ax1.bar(adm_counts.index, adm_counts.values, color=[colors_adm.get(k, "#38bdf8") for k in adm_counts.index], alpha=0.85)
        ax1.set_ylabel("Count", color="#94a3b8")
        ax1.tick_params(colors="#94a3b8")
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.spines[["left", "bottom"]].set_color("#334155")
        st.pyplot(fig1)
        plt.close()

        # ── SHAP GLOBAL ──────────────────────────────────────────────────────
        st.markdown("### 🔍 SHAP — Feature Importance (Global)")
        st.caption("Which features drive anomaly detection the most?")
        shap_sample = X.sample(min(300, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_sample)
        fig2, _ = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor("#0f1729")
        shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols, plot_type="bar", show=False, color="#38bdf8")
        ax2 = plt.gca()
        ax2.set_facecolor("#1e2d4a")
        fig2.patch.set_facecolor("#0f1729")
        ax2.tick_params(colors="#94a3b8")
        ax2.xaxis.label.set_color("#94a3b8")
        st.pyplot(fig2)
        plt.close()

        # ── INDIVIDUAL PATIENT EXPLANATION ──────────────────────────────────
        st.markdown("### 🧑‍⚕️ Explain Individual Patient")
        st.caption("Select a patient to see exactly why they were flagged")
        anomaly_indices = df_model[df_model["Anomaly"] == -1].index.tolist()
        if anomaly_indices:
            selected_idx = st.selectbox("Select anomaly patient:", anomaly_indices[:50])
            patient_row = df_model.loc[[selected_idx]]
            st.write("**Patient details:**")
            st.dataframe(patient_row[[c for c in DISPLAY_COLS if c in patient_row.columns]], use_container_width=True)

            patient_X = X.iloc[[df_model.index.get_loc(selected_idx)]]
            patient_shap = explainer.shap_values(patient_X)

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            fig3.patch.set_facecolor("#0f1729")
            ax3.barh(feature_cols, patient_shap[0], color=["#f87171" if v > 0 else "#38bdf8" for v in patient_shap[0]])
            ax3.set_facecolor("#1e2d4a")
            ax3.set_xlabel("SHAP Value (impact on anomaly score)", color="#94a3b8")
            ax3.set_title(f"Patient #{selected_idx} — Why flagged as anomaly?", color="#f8fafc")
            ax3.tick_params(colors="#94a3b8")
            ax3.spines[["top", "right"]].set_visible(False)
            ax3.spines[["left", "bottom"]].set_color("#334155")
            ax3.axvline(0, color="#475569", linewidth=0.8)
            st.pyplot(fig3)
            plt.close()
            st.markdown("> 🔴 **Red bars** push toward anomaly &nbsp;&nbsp; 🔵 **Blue bars** push toward normal")

        # ── DOWNLOAD RESULTS ─────────────────────────────────────────────────
        st.markdown("### 📥 Download Results")
        csv = df_model[[c for c in DISPLAY_COLS if c in df_model.columns]].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download anomaly results as CSV", csv, "anomaly_results.csv", "text/csv")

        st.markdown("---")
        st.success("✅ Analysis complete!")

else:
    st.info("👆 Click **Detect Anomalies** to run the model")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#64748b; font-size:13px'>XAI-AD · Punitha Karmegam · Master Data Intelligence · ISEP Paris 2025</div>", unsafe_allow_html=True)
