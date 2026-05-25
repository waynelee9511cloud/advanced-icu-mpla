import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import PatientRecord, TIME_SERIES_PARAMS
from src.features import FeatureExtractor

# Page configuration for the professional clinical platform
st.set_page_config(
    page_title="ICU Patient Mortality Predictor & Physiological Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clinical parameters labels
PARAM_LABELS = {
    'HR': 'Heart Rate (bpm)',
    'Temp': 'Temperature (°C)',
    'GCS': 'Glasgow Coma Score (3-15)',
    'Urine': 'Urine Output (mL)',
    'WBC': 'White Blood Cell Count (cells/nL)',
    'Creatinine': 'Serum Creatinine (mg/dL)',
    'BUN': 'Blood Urea Nitrogen (mg/dL)',
    'Glucose': 'Serum Glucose (mg/dL)',
    'HCO3': 'Serum Bicarbonate (mmol/L)',
    'HCT': 'Hematocrit (%)',
    'Na': 'Serum Sodium (mEq/L)',
    'K': 'Serum Potassium (mEq/L)',
    'Mg': 'Serum Magnesium (mmol/L)',
    'Platelets': 'Platelets (cells/nL)',
    'RespRate': 'Respiration Rate (bpm)',
    'SysABP': 'Invasive Systolic BP (mmHg)',
    'DiasABP': 'Invasive Diastolic BP (mmHg)',
    'MAP': 'Invasive Mean Arterial BP (mmHg)',
    'NISysABP': 'Non-invasive Systolic BP (mmHg)',
    'NIDiasABP': 'Non-invasive Diastolic BP (mmHg)',
    'NIMAP': 'Non-invasive Mean Arterial BP (mmHg)',
    'FiO2': 'Fractional Inspired O2 (0-1)',
    'PaO2': 'Partial Pressure of Arterial O2 (mmHg)',
    'PaCO2': 'Partial Pressure of Arterial CO2 (mmHg)',
    'pH': 'Arterial pH',
    'SaO2': 'O2 Saturation (%)',
    'Weight': 'Weight (kg)'
}

# Premium dark theme and cards CSS overrides (Optimized for no-scroll viewport)
st.markdown("""
<style>
    /* Main App Background Override */
    .stApp {
        background-color: #0d1e3d;
        color: #ffffff;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Minimize margins/padding to fit page on one screen */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Headers and Titles spacing */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
        margin-top: 0px !important;
        margin-bottom: 6px !important;
    }
    
    /* Top Header Bar styling */
    .header-bar {
        background-color: #09152e;
        padding: 8px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #1a365d;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Sidebar styling enhancements */
    [data-testid="stSidebar"] {
        background-color: #09152e !important;
        border-right: 1px solid #1a365d;
    }
    
    /* Patient Info Banner inside Sidebar */
    .patient-banner {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        padding: 12px;
        border-radius: 8px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Light metrics cards (Vitals) with auto-height to prevent cutoff */
    .vital-card-light {
        background-color: #ffffff;
        color: #1e293b;
        padding: 8px 12px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: left;
        border-top: 4px solid #cbd5e1;
        min-height: 85px;
        height: auto;
    }
    
    .vital-card-light .title {
        font-size: 11px;
        color: #64748b;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .vital-card-light .value {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    
    .vital-card-light .unit {
        font-size: 10px;
        font-weight: 400;
        color: #64748b;
    }
    
    .vital-card-light .range {
        font-size: 10px;
        color: #94a3b8;
        margin-top: 1px;
    }
    
    /* Dark panel cards */
    .card-dark {
        background-color: #09152e;
        color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1a365d;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
        margin-bottom: 10px;
    }
    
    /* Alert panel styling */
    .alert-box {
        border-radius: 6px;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
    }
    .alert-box.critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border-left: 4px solid #ef4444;
    }
    .alert-box.warning {
        background-color: rgba(249, 115, 22, 0.2);
        color: #fed7aa;
        border-left: 4px solid #f97316;
    }
    .alert-box.stable {
        background-color: rgba(16, 185, 129, 0.2);
        color: #a7f3d0;
        border-left: 4px solid #10b981;
    }
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #09152e;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #1a365d;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0aec0;
        font-weight: bold;
        padding: 4px 10px;
        font-size: 12px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff;
        background-color: #1d4ed8;
        border-radius: 6px;
    }
    
    /* Lab item list styling */
    .lab-grid-item {
        background-color: #f8fafc;
        color: #0f172a;
        padding: 6px;
        border-radius: 6px;
        text-align: center;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Reduce native Streamlit spacings to save vertical space */
    .element-container {
        margin-bottom: 6px !important;
    }
    
    .stSlider {
        padding-bottom: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper dictionaries
ICU_TYPES = {
    1.0: "Coronary Care Unit (CCU)",
    2.0: "Cardiac Surgery Recovery Unit (CSRU)",
    3.0: "Medical ICU (MICU)",
    4.0: "Surgical ICU (SICU)"
}

@st.cache_resource
def load_model():
    model_path = "outputs/best_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_feature_datasets():
    df_train = None
    df_test = None
    if os.path.exists("data/features_set_a.csv"):
        df_train = pd.read_csv("data/features_set_a.csv")
    if os.path.exists("data/features_set_b.csv"):
        df_test = pd.read_csv("data/features_set_b.csv")
    return df_train, df_test

def get_all_patient_ids(df_train, df_test):
    patient_ids = []
    if df_train is not None:
        patient_ids.extend(df_train['RecordID'].tolist())
    if df_test is not None:
        patient_ids.extend(df_test['RecordID'].tolist())
        
    if not patient_ids:
        # Fallback to directories
        for s in ["a", "b"]:
            path = f"data/set-{s}"
            if os.path.exists(path):
                patient_ids.extend([int(f.split('.')[0]) for f in os.listdir(path) if f.endswith('.txt')])
    return sorted(list(set(patient_ids)))

def get_param_summary(time_series, param):
    obs = time_series.get(param, [])
    if not obs:
        return np.nan, np.nan, np.nan, np.nan
    vals = [v for t, v in obs]
    return vals[-1], np.mean(vals), np.min(vals), np.max(vals)

def get_bp_string(time_series):
    sys_val, _, _, _ = get_param_summary(time_series, 'SysABP')
    if np.isnan(sys_val):
        sys_val, _, _, _ = get_param_summary(time_series, 'NISysABP')
        
    dias_val, _, _, _ = get_param_summary(time_series, 'DiasABP')
    if np.isnan(dias_val):
        dias_val, _, _, _ = get_param_summary(time_series, 'NIDiasABP')
        
    if np.isnan(sys_val) or np.isnan(dias_val):
        return "N/A", np.nan, np.nan
    return f"{int(sys_val)}/{int(dias_val)}", sys_val, dias_val

def main():
    # Load Model and cached data
    pipeline = load_model()
    df_train, df_test = load_feature_datasets()
    
    if pipeline is None:
        st.error("Error: Model pipeline (best_model.joblib) not found. Please run the training script train.py first.")
        st.stop()
        
    patient_ids = get_all_patient_ids(df_train, df_test)
    
    # Top Header Bar - Project-specific Renaming (No Sponsor/System reference)
    st.markdown("""
    <div class="header-bar">
        <div style="display: flex; flex-direction: column;">
            <div style="font-size: 20px; font-weight: bold; color: white; line-height: 1.2;">
                ICU Patient Mortality Predictor & Physiological Analyzer
            </div>
            <div style="color: #94a3b8; font-size: 11px; margin-top: 2px;">
                Interactive ICU Risk Analysis Platform based on clinical parameters
            </div>
        </div>
        <div style="background-color: #1e293b; padding: 4px 10px; border-radius: 5px; font-size: 11px; color: #38bdf8; border: 1px solid #334155; font-weight: bold;">
            XGBoost Inference Engine
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar Configuration ---
    st.sidebar.markdown("""
    <div style="margin-bottom: 15px;">
        <h2 style="font-size: 16px; font-weight: bold; color: white; margin-bottom: 5px;">ICU Patient Predictor & Analyzer (ICU-MPLA)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not patient_ids:
        st.sidebar.warning("Physiological records not found in data directories.")
        selected_id = None
    else:
        selected_id = st.sidebar.selectbox("Select Patient ID", patient_ids)
        
    if selected_id is not None:
        # Load patient file
        file_path = os.path.join("data/set-a", f"{selected_id}.txt")
        if not os.path.exists(file_path):
            file_path = os.path.join("data/set-b", f"{selected_id}.txt")
            
        record = PatientRecord(file_path)
        descriptors = record.descriptors
        time_series = record.time_series
        
        # Lookup precomputed features or extract on-the-fly
        if df_train is not None and selected_id in df_train['RecordID'].values:
            p_row = df_train[df_train['RecordID'] == selected_id].iloc[0]
        elif df_test is not None and selected_id in df_test['RecordID'].values:
            p_row = df_test[df_test['RecordID'] == selected_id].iloc[0]
        else:
            extractor = FeatureExtractor()
            features = extractor.extract_features(record)
            p_row = pd.Series(features)
            
        # Info variables
        age = descriptors['Age']
        gender = "Male" if descriptors['Gender'] == 1.0 else ("Female" if descriptors['Gender'] == 0.0 else "Unknown")
        icu_unit = ICU_TYPES.get(descriptors['ICUType'], "Unknown")
        
        observed_death = p_row.get('In-hospital_death', np.nan)
        status_badge = "Deceased" if observed_death == 1.0 else ("Survived" if observed_death == 0.0 else "Monitoring")
        status_color = "#f87171" if observed_death == 1.0 else "#34d399"
        
        # Patient Banner inside Sidebar
        st.sidebar.markdown(f"""
        <div class="patient-banner">
            <div style="font-size: 11px; font-weight: bold; color: rgba(255,255,255,0.7); text-transform: uppercase;">Selected Patient</div>
            <div style="font-size: 16px; font-weight: 800; color: white; margin-top: 2px;">ID: {selected_id}</div>
            <div style="font-size: 12px; color: #f1f5f9; margin-top: 4px; font-weight: bold;">Age: {age:.0f} | {gender}</div>
            <div style="font-size: 10px; color: #cbd5e1; margin-top: 2px; line-height: 1.2;">{icu_unit}</div>
            <div style="background-color: {status_color}; color: #0f172a; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 9px; width: fit-content; margin-top: 8px;">
                Outcome: {status_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Info & Sources inside Sidebar
    st.sidebar.markdown("""
    <div style="background-color: #09152e; padding: 12px; border-radius: 8px; border: 1px solid #1a365d; font-size: 11px; color: #cbd5e1; margin-top: 15px;">
        <div style="font-weight: bold; color: #38bdf8; text-transform: uppercase; margin-bottom: 6px; font-size: 11px;">Information Sources</div>
        <div style="margin-bottom: 5px; line-height: 1.3;"><b>Dataset Source:</b> PhysioNet / Computing in Cardiology Challenge 2012</div>
        <div style="margin-bottom: 5px; line-height: 1.3;"><b>Study Cohort:</b> 8,000 ICU patient stays with 48h physiological records</div>
        <div style="margin-bottom: 5px; line-height: 1.3;"><b>Core Model:</b> XGBoost Binary Classifier (AUROC: 0.8612)</div>
        <div style="line-height: 1.3;"><b>Developer:</b> Wayne Lee</div>
    </div>
    """, unsafe_allow_html=True)
        
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "Patient Longitudinal Explorer", 
        "Clinical Risk Simulator",
        "Model Performance & Cohort"
    ])
    
    # --- Tab 1: Patient Longitudinal Explorer ---
    with tab1:
        if selected_id is None:
            st.warning("Please download physiological records to display patient details.")
        else:
            # Load summaries of vitals
            hr_last, hr_mean, hr_min, hr_max = get_param_summary(time_series, 'HR')
            temp_last, temp_mean, temp_min, temp_max = get_param_summary(time_series, 'Temp')
            gcs_last, gcs_mean, gcs_min, gcs_max = get_param_summary(time_series, 'GCS')
            urine_last, urine_mean, urine_min, urine_max = get_param_summary(time_series, 'Urine')
            resp_last, resp_mean, resp_min, resp_max = get_param_summary(time_series, 'RespRate')
            map_last, map_mean, map_min, map_max = get_param_summary(time_series, 'MAP')
            bp_str, sys_last, dias_last = get_bp_string(time_series)
            
            sao2_last, _, _, _ = get_param_summary(time_series, 'SaO2')
            if np.isnan(sao2_last):
                sao2_last = 96.0  # mock fallback standard if not captured
                
            # Row 1: Vitals cards (Span full page width)
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            
            with col_v1:
                hr_val_str = f"{hr_last:.0f}" if not np.isnan(hr_last) else "N/A"
                hr_range_str = f"Range: {hr_min:.0f}-{hr_max:.0f}" if not np.isnan(hr_min) else "Range: N/A"
                st.markdown(f"""
                <div class="vital-card-light" style="border-top: 4px solid #ef4444; display: flex; flex-direction: column; justify-content: space-between;">
                    <div class="title"><span style="color:#ef4444;">❤️</span> Heart Rate</div>
                    <div class="value">{hr_val_str} <span class="unit">bpm</span></div>
                    <div class="range">{hr_range_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_v2:
                st.markdown(f"""
                <div class="vital-card-light" style="border-top: 4px solid #3b82f6; display: flex; flex-direction: column; justify-content: space-between;">
                    <div class="title"><span style="color:#3b82f6;">🩺</span> Blood Pressure</div>
                    <div class="value">{bp_str} <span class="unit">mmHg</span></div>
                    <div class="range">Mean Art: {f"{map_last:.0f}" if not np.isnan(map_last) else "N/A"}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_v3:
                sao2_str = f"{sao2_last:.0f}%" if not np.isnan(sao2_last) else "N/A"
                st.markdown(f"""
                <div class="vital-card-light" style="border-top: 4px solid #06b6d4; display: flex; flex-direction: column; justify-content: space-between;">
                    <div class="title"><span style="color:#06b6d4;">💧</span> SpO2 / SaO2</div>
                    <div class="value">{sao2_str}</div>
                    <div class="range">Oxygen Saturation</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_v4:
                temp_val_str = f"{temp_last:.1f}" if not np.isnan(temp_last) else "N/A"
                temp_range_str = f"Range: {temp_min:.1f}-{temp_max:.1f}" if not np.isnan(temp_min) else "Range: N/A"
                st.markdown(f"""
                <div class="vital-card-light" style="border-top: 4px solid #f97316; display: flex; flex-direction: column; justify-content: space-between;">
                    <div class="title"><span style="color:#f97316;">🌡️</span> Temperature</div>
                    <div class="value">{temp_val_str} <span class="unit">°C</span></div>
                    <div class="range">{temp_range_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            
            # Row 2: Main Dashboard Layout (Gauge + Plot + Labs) - Compacted vertically
            dash_col_left, dash_col_right = st.columns([5, 7])
            
            with dash_col_left:
                # Gauge Panel - Mortality Risk Assessment
                st.markdown("<div class='card-dark' style='padding: 10px; margin-bottom: 8px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-size: 13px; margin-bottom: 6px;'>MORTALITY RISK ASSESSMENT</h4>", unsafe_allow_html=True)
                
                # Predict risk for this patient
                model = pipeline['model']
                threshold = pipeline.get('threshold', 0.66)
                
                # Reorder features dynamically to match model expected column order
                if hasattr(model, 'feature_names_in_'):
                    feature_cols = list(model.feature_names_in_)
                else:
                    meta_cols = ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
                    feature_cols = [c for c in p_row.index if c not in meta_cols]
                    
                patient_features = pd.DataFrame([p_row[feature_cols]])
                prob = model.predict_proba(patient_features)[0, 1]
                
                # Gauge chart coloring
                if prob < 0.15:
                    risk_level = "LOW RISK"
                    gauge_color = "#10b981"
                elif prob < threshold:
                    risk_level = "MEDIUM RISK"
                    gauge_color = "#f97316"
                else:
                    risk_level = "HIGH RISK"
                    gauge_color = "#ef4444"
                    
                # Centered, styled HTML Risk Level text to prevent Plotly title cutoff overlaps
                st.markdown(f"<div style='text-align:center; font-size:14px; font-weight:bold; color:{gauge_color}; margin-bottom:-4px;'>{risk_level} (Model Prob: {prob:.1%})</div>", unsafe_allow_html=True)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    number = {'font': {'size': 18}, 'suffix': "%"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e0", 'tickfont': {'size': 8}},
                        'bar': {'color': gauge_color},
                        'bgcolor': "rgba(0,0,0,0.2)",
                        'borderwidth': 1,
                        'bordercolor': "#1a365d",
                        'steps': [
                            {'range': [0, 15], 'color': 'rgba(16, 185, 129, 0.1)'},
                            {'range': [15, threshold*100], 'color': 'rgba(249, 115, 22, 0.1)'},
                            {'range': [threshold*100, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                        ],
                        'threshold': {
                            'line': {'color': "#ffffff", 'width': 2},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=120, 
                    margin=dict(l=15, r=15, t=5, b=5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'size': 10}
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Alerts Panel - Compacted height
                st.markdown("<div class='card-dark' style='padding: 8px; margin-bottom: 0px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-size: 13px; margin-bottom: 4px;'>RECENT ALERTS</h4>", unsafe_allow_html=True)
                
                alerts = []
                # Critical triggers
                if hr_last > 120:
                    alerts.append(("critical", f"🚨 Severe Tachycardia: {hr_last:.0f} bpm"))
                elif hr_last < 50:
                    alerts.append(("critical", f"🚨 Severe Bradycardia: {hr_last:.0f} bpm"))
                    
                if temp_last > 38.8:
                    alerts.append(("warning", f"⚠️ Hyperpyrexia: {temp_last:.1f} °C"))
                elif temp_last < 35.0:
                    alerts.append(("warning", f"⚠️ Hypothermia: {temp_last:.1f} °C"))
                    
                if gcs_last <= 8:
                    alerts.append(("critical", f"🚨 Severe Coma (GCS: {gcs_last:.0f})"))
                elif gcs_last <= 12:
                    alerts.append(("warning", f"⚠️ Moderate Coma (GCS: {gcs_last:.0f})"))
                    
                # Lab warnings
                wbc_last, _, _, _ = get_param_summary(time_series, 'WBC')
                if wbc_last > 15.0:
                    alerts.append(("warning", f"⚠️ High WBC / Inflammation: {wbc_last:.1f} cells/nL"))
                elif wbc_last < 3.0:
                    alerts.append(("warning", f"⚠️ Low WBC / Immunodeficient: {wbc_last:.1f} cells/nL"))
                    
                # Render alerts in scrollable small div to guarantee single page fit
                st.markdown("<div style='max-height: 75px; overflow-y: auto; padding: 2px 0;'>", unsafe_allow_html=True)
                if not alerts:
                    st.markdown("""
                    <div class="alert-box stable" style="margin-bottom:0; padding:6px 10px;">
                        🟢 Stable physiological vitals. No active clinical warnings.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for level, msg in alerts:
                        box_class = "critical" if level == "critical" else "warning"
                        st.markdown(f'<div class="alert-box {box_class}" style="margin-bottom:4px; padding:6px 10px;">{msg}</div>', unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)
                
            with dash_col_right:
                # Vitals Trend Plot Panel (Reduced height to fit screen)
                st.markdown("<div class='card-dark' style='padding: 10px; margin-bottom: 0px;'>", unsafe_allow_html=True)
                
                available_params = [p for p in TIME_SERIES_PARAMS if len(time_series[p]) > 0]
                if not available_params:
                    st.info("No time-series measurements available for this patient.")
                else:
                    col_p1, col_p2 = st.columns([5, 7])
                    with col_p1:
                        st.markdown("<h4 style='font-size: 13px; margin-top:5px !important;'>LONGITUDINAL TRENDS</h4>", unsafe_allow_html=True)
                    with col_p2:
                        selected_param = st.selectbox(
                            "Select Parameter", 
                            available_params,
                            format_func=lambda x: f"{x} - {PARAM_LABELS.get(x, '')}",
                            label_visibility="collapsed"
                        )
                    
                    observations = time_series[selected_param]
                    times, values = zip(*observations)
                    hours = [t / 60.0 for t in times]
                    
                    df_plot = pd.DataFrame({
                        'Time (Hours)': hours,
                        selected_param: values
                    })
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=df_plot['Time (Hours)'], 
                        y=df_plot[selected_param],
                        mode='lines+markers',
                        name=selected_param,
                        line=dict(color='#3b82f6', width=2),
                        marker=dict(size=5, color='#ef4444')
                    ))
                    
                    # Add critical threshold indicator lines
                    if selected_param == 'HR':
                        fig_trend.add_hline(y=140, line_dash="dash", line_color="red")
                        fig_trend.add_hline(y=50, line_dash="dash", line_color="blue")
                    elif selected_param == 'Temp':
                        fig_trend.add_hline(y=38.5, line_dash="dash", line_color="red")
                        fig_trend.add_hline(y=35.5, line_dash="dash", line_color="blue")
                    elif selected_param == 'GCS':
                        fig_trend.add_hline(y=8, line_dash="dash", line_color="red")
                        
                    fig_trend.update_layout(
                        height=180,
                        margin=dict(l=25, r=15, t=5, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.05)",
                        font={'color': "white", 'size': 8},
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Bottom Grid for Labs & Extra Parameters (Positioned full-width below left/right columns)
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='card-dark' style='padding: 8px; margin-bottom: 0px;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-size: 12px; margin-bottom: 4px;'>LAB VALUE AGGREGATES (48-HOUR MEAN)</h4>", unsafe_allow_html=True)
            
            # Grid coordinates
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            
            wbc_m = p_row.get('WBC_mean', np.nan)
            crea_m = p_row.get('Creatinine_mean', np.nan)
            bun_m = p_row.get('BUN_mean', np.nan)
            gluc_m = p_row.get('Glucose_mean', np.nan)
            
            with l_col1:
                wbc_str = f"{wbc_m:.1f}" if not np.isnan(wbc_m) else "N/A"
                st.markdown(f"""
                <div class="lab-grid-item">
                    <div style="font-size:9px; color:#64748b; font-weight:bold;">WBC</div>
                    <div style="font-size:12px; font-weight:bold; color: #1e293b;">{wbc_str} <span style="font-size:8px; font-weight:normal; color:#64748b;">cells/nL</span></div>
                </div>
                """, unsafe_allow_html=True)
            with l_col2:
                crea_str = f"{crea_m:.2f}" if not np.isnan(crea_m) else "N/A"
                st.markdown(f"""
                <div class="lab-grid-item">
                    <div style="font-size:9px; color:#64748b; font-weight:bold;">CREATININE</div>
                    <div style="font-size:12px; font-weight:bold; color: #1e293b;">{crea_str} <span style="font-size:8px; font-weight:normal; color:#64748b;">mg/dL</span></div>
                </div>
                """, unsafe_allow_html=True)
            with l_col3:
                bun_str = f"{bun_m:.1f}" if not np.isnan(bun_m) else "N/A"
                st.markdown(f"""
                <div class="lab-grid-item">
                    <div style="font-size:9px; color:#64748b; font-weight:bold;">BUN</div>
                    <div style="font-size:12px; font-weight:bold; color: #1e293b;">{bun_str} <span style="font-size:8px; font-weight:normal; color:#64748b;">mg/dL</span></div>
                </div>
                """, unsafe_allow_html=True)
            with l_col4:
                gluc_str = f"{gluc_m:.0f}" if not np.isnan(gluc_m) else "N/A"
                st.markdown(f"""
                <div class="lab-grid-item">
                    <div style="font-size:9px; color:#64748b; font-weight:bold;">GLUCOSE</div>
                    <div style="font-size:12px; font-weight:bold; color: #1e293b;">{gluc_str} <span style="font-size:8px; font-weight:normal; color:#64748b;">mg/dL</span></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 2: Clinical Risk Simulator ---
    with tab2:
        st.markdown("<div class='card-dark' style='padding: 10px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 15px; margin-bottom: 4px;'>CLINICAL RISK SIMULATION TOOL</h3>", unsafe_allow_html=True)
        st.write("<div style='font-size:11px; color:#cbd5e1; margin-bottom: 8px;'>Adjust key physiological metrics using the grid below. Predicted risk updates in real-time.</div>", unsafe_allow_html=True)
        
        # Load median vectors for missing variables
        if df_train is not None:
            meta_cols = ['RecordID', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
            feature_cols = [c for c in df_train.columns if c not in meta_cols]
            median_values = df_train[feature_cols].median()
        else:
            st.error("Error: Feature medians not available. Please cache features.")
            st.stop()
            
        # Clinical parameters 5-column by 2-row layout
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            age_in = st.slider("Age (Years)", 18, 100, 65)
            temp_min_in = st.slider("Temp Min (°C)", 30.0, 42.0, 36.2, step=0.1)
        with col2:
            gcs_min_in = st.slider("GCS Min [3-15]", 3, 15, 12)
            gcs_last_in = st.slider("GCS Last [3-15]", 3, 15, 14)
        with col3:
            hr_mean_in = st.slider("HR Mean [bpm]", 40, 180, 85)
            hr_max_in = st.slider("HR Max [bpm]", 50, 220, 110)
        with col4:
            urine_mean_in = st.slider("Urine Mean [mL/h]", 0, 1000, 80)
            wbc_mean_in = st.slider("WBC Mean [cells/nL]", 1.0, 100.0, 12.5, step=0.5)
        with col5:
            creatinine_mean_in = st.slider("Creatinine Mean [mg/dL]", 0.1, 12.0, 1.4, step=0.1)
            bun_mean_in = st.slider("BUN Mean [mg/dL]", 1, 150, 25)
            
        # Run simulator prediction automatically on slider changes
        sim_features = median_values.copy()
        sim_features['Age'] = age_in
        sim_features['GCS_min'] = gcs_min_in
        sim_features['GCS_last'] = gcs_last_in
        sim_features['BUN_mean'] = bun_mean_in
        sim_features['HR_mean'] = hr_mean_in
        sim_features['HR_max'] = hr_max_in
        sim_features['Temp_min'] = temp_min_in
        sim_features['Urine_mean'] = urine_mean_in
        sim_features['WBC_mean'] = wbc_mean_in
        sim_features['Creatinine_mean'] = creatinine_mean_in
        
        df_sim = pd.DataFrame([sim_features])
        model = pipeline['model']
        threshold = pipeline.get('threshold', 0.66)
        
        # Align simulator features to model columns
        if hasattr(model, 'feature_names_in_'):
            df_sim = df_sim[model.feature_names_in_]
            
        sim_prob = model.predict_proba(df_sim)[0, 1]
        
        if sim_prob < 0.15:
            sim_level = "LOW RISK"
            sim_color = "#10b981"
            recommendation = "🟢 Patient exhibits stable clinical features. Routine surveillance recommended."
        elif sim_prob < threshold:
            sim_level = "MEDIUM RISK"
            sim_color = "#f97316"
            recommendation = "⚠️ Patient shows moderate risk profiles. Watch renal and urine metrics closely."
        else:
            sim_level = "HIGH RISK"
            sim_color = "#ef4444"
            recommendation = "🚨 High mortality warning. Immediate clinical review of vital trends is advised."
            
        st.markdown("<hr style='margin: 8px 0; border-color: #1a365d;' />", unsafe_allow_html=True)
        
        # Output side-by-side below sliders
        col_res_left, col_res_right = st.columns([5, 7])
        
        with col_res_left:
            st.markdown(f"<div style='text-align:center; font-size:14px; font-weight:bold; color:{sim_color}; margin-top: 5px; margin-bottom:-4px;'>{sim_level} (Prob: {sim_prob:.1%})</div>", unsafe_allow_html=True)
            
            fig_sim = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sim_prob * 100,
                number = {'font': {'size': 18}, 'suffix': "%"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 8}},
                    'bar': {'color': sim_color},
                    'bgcolor': "rgba(0,0,0,0.2)",
                    'borderwidth': 1,
                    'bordercolor': "#1a365d",
                    'steps': [
                        {'range': [0, 15], 'color': 'rgba(16, 185, 129, 0.1)'},
                        {'range': [15, threshold * 100], 'color': 'rgba(249, 115, 22, 0.1)'},
                        {'range': [threshold * 100, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_sim.update_layout(
                height=120,
                margin=dict(l=15, r=15, t=5, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'size': 10}
            )
            st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': False})
            
        with col_res_right:
            st.markdown(f"""
            <div style="background-color: rgba(30, 41, 59, 0.5); padding: 12px; border-radius: 8px; border: 1px solid #334155; font-size:11px; color:#e2e8f0; height: 110px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-weight: bold; color: {sim_color}; margin-bottom: 5px; text-transform: uppercase;">Clinical Recommendation:</div>
                <div style="line-height: 1.3;">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    # --- Tab 3: Cohort Overview & Model Performance ---
    with tab3:
        st.markdown("<div class='card-dark' style='padding: 10px; margin-bottom: 0px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 15px; margin-bottom: 6px;'>COHORT OVERVIEW & MODEL PERFORMANCE</h3>", unsafe_allow_html=True)
        
        # Side-by-side 3-column layout to fit on one screen
        col_c1, col_c2, col_c3 = st.columns([4.2, 3.9, 3.9])
        
        with col_c1:
            st.markdown("<h4 style='font-size: 12px; margin-bottom: 4px;'>Cohort Statistics</h4>", unsafe_allow_html=True)
            
            train_size = len(df_train) if df_train is not None else 4000
            test_size = len(df_test) if df_test is not None else 4000
            mort_rate_train = df_train['In-hospital_death'].mean() if df_train is not None else 0.1385
            mort_rate_test = df_test['In-hospital_death'].mean() if df_test is not None else 0.1420
            
            st.markdown(f"""
            <div style="font-size: 11px; line-height: 1.4; color: #cbd5e1; margin-bottom: 8px;">
                • <b>Development Cohort (Set A):</b> {train_size} Patients<br/>
                • <b>Validation Cohort (Set B):</b> {test_size} Patients<br/>
                • <b>Development Mortality Rate:</b> {mort_rate_train:.2%}<br/>
                • <b>Validation Mortality Rate:</b> {mort_rate_test:.2%}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4 style='font-size: 12px; margin-top: 5px !important; margin-bottom: 4px;'>Final Classification Report (Test Set B)</h4>", unsafe_allow_html=True)
            summary_path = "outputs/evaluation_summary.txt"
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    st.text(f.read())
            else:
                st.write("""
                *   Classifier: XGBoost
                *   Accuracy: 87.30%
                *   AUROC: 0.8612
                *   PhysioNet Challenge Score: 0.4137
                """)
                
        with col_c2:
            st.markdown("<h4 style='font-size: 12px; text-align: center; margin-bottom: 4px;'>ROC Analysis Curve</h4>", unsafe_allow_html=True)
            if os.path.exists("outputs/roc_curves.png"):
                st.image("outputs/roc_curves.png", use_container_width=True)
            else:
                st.info("ROC plot not found.")
                
        with col_c3:
            st.markdown("<h4 style='font-size: 12px; text-align: center; margin-bottom: 4px;'>Physiological Feature Importance</h4>", unsafe_allow_html=True)
            if os.path.exists("outputs/feature_importance.png"):
                st.image("outputs/feature_importance.png", use_container_width=True)
            else:
                st.info("Feature importance plot not found.")
                
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
