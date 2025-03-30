import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AusCycling Performance Pro")

# --- User Profile Section (Top-Right) ---
with st.sidebar:
    st.header("User Profile")
    profile_tab1, profile_tab2 = st.tabs(["My Profile", "Settings"])
    
    with profile_tab1:
        # Profile picture upload
        profile_pic = st.file_uploader("Profile Photo", type=["jpg", "png"])
        if profile_pic:
            img = Image.open(profile_pic)
            st.image(img, width=100)
        
        # Bio info
        st.text_input("Name", key="profile_name")
        st.text_input("Email", key="profile_email")
        st.text_input("Phone", key="profile_phone")
    
    with profile_tab2:
        st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.slider("Data Precision", 1, 10, 5, key="data_precision")

# --- Main App Tabs ---
tab1, tab2, tab3 = st.tabs(["Data Input", "Performance Analysis", "Optimization"])

with tab1:
    st.header("New Data Entry")
    
    # Simplified input form (no rider selection required)
    with st.form("new_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            rider_name = st.text_input("Rider Name")
            weight = st.number_input("Weight (kg)", min_value=40, max_value=120, value=70)
            power = st.number_input("Power Output (W)", min_value=100, max_value=2000, value=250)
        
        with col2:
            cda = st.number_input("Drag Coefficient (CdA)", min_value=0.1, max_value=0.4, value=0.25)
            rr = st.number_input("Rolling Resistance", min_value=0.002, max_value=0.01, value=0.004)
            slope = st.slider("Track Slope (%)", min_value=-5.0, max_value=5.0, value=0.0)
        
        if st.form_submit_button("Save Data"):
            # Save to session state or database
            st.success("Data saved!")

with tab2:
    st.header("Performance Analysis")
    
    # Visualization options
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Power Curve", "Fatigue Analysis", "Energy Expenditure"],
        key="analysis_type"
    )
    
    if analysis_type == "Power Curve":
        # Your existing power curve visualization with Plotly
        t = np.linspace(1, 100, 100000)
        fig = px.line(title="Power Curve Analysis")
        
        # Add rider data to plot
        # (You would replace this with your actual data)
        fig.add_scatter(x=t, y=250 + 1500/t, name="Sample Rider 1")
        fig.add_scatter(x=t, y=300 + 1200/t, name="Sample Rider 2")
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Optimization Model")
    
    model_type = st.radio(
        "Select Model",
        ["Light Optimization", "Pro Optimization"],
        horizontal=True
    )
    
    if model_type == "Light Optimization":
        st.write("Basic optimization parameters")
        # Add light optimization controls
    else:
        st.write("Advanced optimization parameters")
        # Add pro optimization controls

# --- File Upload Section ---
st.sidebar.header("Data Import")
uploaded_file = st.sidebar.file_uploader(
    "Upload Athlete Data",
    type=["csv"],
    help="Upload CSV with columns: Name, CP, W, Pmax"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['athlete_data'] = df
    st.sidebar.success("Data loaded successfully!")