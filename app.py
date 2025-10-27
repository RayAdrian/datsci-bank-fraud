
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("merged_bank_bureau_data.csv")
    return df

df = load_data()

st.title("📊 Synthetic Bank Dashboard")
st.markdown("Visual exploration of application and credit data.")

# --- KPI Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Approval Rate", f"{100 * (1 - df['target'].mean()):.1f}%")
with col2:
    st.metric("Fraud Rate", f"{100 * df['fraud_flag'].mean():.1f}%")
with col3:
    st.metric("Avg. Income", f"₱{df['monthly_income'].mean():,.0f}")
with col4:
    st.metric("Avg. Utilization", f"{df['utilization_ratio'].mean():.2f}")

st.markdown("---")

# --- Filters ---
with st.sidebar:
    st.header("🔍 Filters")
    region = st.selectbox("Region", options=["All"] + sorted(df['region'].dropna().unique().tolist()))
    employment = st.selectbox("Employment Type", options=["All"] + sorted(df['employment_type'].dropna().unique().tolist()))
    risk = st.selectbox("Risk Grade", options=["All"] + sorted(df['risk_grade'].dropna().unique().tolist()))

# Apply filters
filtered_df = df.copy()
if region != "All":
    filtered_df = filtered_df[filtered_df["region"] == region]
if employment != "All":
    filtered_df = filtered_df[filtered_df["employment_type"] == employment]
if risk != "All":
    filtered_df = filtered_df[filtered_df["risk_grade"] == risk]

# --- Visualizations ---

st.subheader("🎯 Approval by Demographics")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x="gender", color="target", barmode="group", title="Approval by Gender")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(filtered_df, x="marital_status", color="target", barmode="group", title="Approval by Marital Status")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("💼 Employment & Credit Behavior")

fig3 = px.box(filtered_df, x="employment_type", y="monthly_income", color="target", title="Income by Employment Type")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.scatter(filtered_df, x="utilization_ratio", y="payment_history_on_time_ratio",
                  color="target", hover_data=["application_id"],
                  title="Utilization vs On-Time Payment History")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("📉 Risk Grade & Defaults")
fig5 = px.histogram(filtered_df, x="risk_grade", color="target", barmode="group", title="Target Distribution by Risk Grade")
st.plotly_chart(fig5, use_container_width=True)

st.subheader("📍 Regional View")
fig6 = px.histogram(filtered_df, x="region", color="target", barmode="group", title="Applications by Region")
st.plotly_chart(fig6, use_container_width=True)

# --------------------------
# 📈 Correlation Heatmap
# --------------------------
st.subheader("📊 Correlation Heatmap")

numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64'])
corr = numeric_cols.corr().round(2)

heatmap = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    colorscale='Viridis',
    showscale=True,
    reversescale=True
)

heatmap.update_layout(height=800, margin=dict(l=40, r=40, t=40, b=40))
st.plotly_chart(heatmap, use_container_width=True)

# --------------------------
# 🕵️ Fraud Rate by Segment
# --------------------------
st.subheader("🕵️ Fraud Risk by Segment")

col1, col2 = st.columns(2)

with col1:
    fraud_by_channel = filtered_df.groupby("application_channel")["fraud_flag"].mean().reset_index()
    fig_fraud_channel = px.bar(fraud_by_channel, x="application_channel", y="fraud_flag",
                               title="Fraud Rate by Application Channel",
                               labels={"fraud_flag": "Fraud Rate"})
    st.plotly_chart(fig_fraud_channel, use_container_width=True)

with col2:
    fraud_by_device = filtered_df.groupby("device_type")["fraud_flag"].mean().reset_index()
    fig_fraud_device = px.bar(fraud_by_device, x="device_type", y="fraud_flag",
                              title="Fraud Rate by Device Type",
                              labels={"fraud_flag": "Fraud Rate"})
    st.plotly_chart(fig_fraud_device, use_container_width=True)

# --------------------------
# 📉 Defaults by Risk Indicators
# --------------------------
st.subheader("📉 Defaults by Risk Indicators")

fig_delinquency = px.histogram(filtered_df, x="recent_delinquency_flag", color="target",
                               barmode="group", title="Defaults vs Recent Delinquency")
st.plotly_chart(fig_delinquency, use_container_width=True)

fig_risk_grade = px.histogram(filtered_df, x="risk_grade", color="recent_delinquency_flag",
                              barmode="group", title="Delinquency Flag by Risk Grade")
st.plotly_chart(fig_risk_grade, use_container_width=True)

# --------------------------
# 📨 Applications by Channel
# --------------------------
st.subheader("📨 Applications by Channel")

fig_channel = px.histogram(filtered_df, x="application_channel", color="target",
                           barmode="group", title="Target by Application Channel")
st.plotly_chart(fig_channel, use_container_width=True)
