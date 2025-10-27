
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

# Create descriptive labels for better legend readability
df['target_label'] = df['target'].map({0: 'Approved', 1: 'Rejected/Default'})
df['delinquency_label'] = df['recent_delinquency_flag'].map({0: 'No', 1: 'Yes'})

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

# Ensure descriptive labels are present in filtered data
filtered_df['target_label'] = filtered_df['target'].map({0: 'Approved', 1: 'Rejected/Default'})
filtered_df['delinquency_label'] = filtered_df['recent_delinquency_flag'].map({0: 'No', 1: 'Yes'})

# --- Key Insights with Visualizations ---
st.header("🧠 Key Insights")

# --------------------------
# 🔹 Insight 1: High Approval Rates
# --------------------------
st.subheader("🔹 Insight 1: Approval rates are very high (~91.5%), even among higher-risk applicants")
st.markdown("""
- Applicants with poor credit history, low payment ratios, and recent delinquencies are often still approved.
- Risk grades D and E are overrepresented in the "Approved" group.
""")

fig_risk_grade = px.histogram(filtered_df, x="risk_grade", color="target_label", barmode="group",
                              title="Target Distribution by Risk Grade")
st.plotly_chart(fig_risk_grade, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig_gender = px.histogram(filtered_df, x="gender", color="target_label", barmode="group",
                              title="Approval by Gender")
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    fig_marital = px.histogram(filtered_df, x="marital_status", color="target_label", barmode="group",
                               title="Approval by Marital Status")
    st.plotly_chart(fig_marital, use_container_width=True)

st.markdown("---")

# --------------------------
# 🔹 Insight 2: Credit Utilization
# --------------------------
st.subheader("🔹 Insight 2: Credit utilization is strongly linked to repayment performance")
st.markdown("""
- Higher utilization ratio correlates with lower on-time payment history and higher delinquency.
- Yet, utilization is not currently treated as a disqualifier.
""")

fig_utilization = px.scatter(filtered_df, x="utilization_ratio", y="payment_history_on_time_ratio",
                             color="target_label", hover_data=["application_id"],
                             title="Utilization vs On-Time Payment History")
st.plotly_chart(fig_utilization, use_container_width=True)

st.markdown("---")

# --------------------------
# 🔹 Insight 3: Fraud in Digital Channels
# --------------------------
st.subheader("🔹 Insight 3: Fraud patterns concentrate in digital channels")
st.markdown("""
- Fraud rate is highest among applications submitted via Online and Unknown devices.
- Fraudulent profiles often have incomplete contact information (invalid email/phone) and short employment history.
""")

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

fig_channel = px.histogram(filtered_df, x="application_channel", color="target_label",
                           barmode="group", title="Target by Application Channel")
st.plotly_chart(fig_channel, use_container_width=True)

st.markdown("---")

# --------------------------
# 🔹 Insight 4: Self-Employed Risk
# --------------------------
st.subheader("🔹 Insight 4: Self-employed applicants exhibit higher income variance and risk")
st.markdown("""
- Though income can be high, the group shows inconsistent bureau patterns and more closed accounts.
""")

fig_income = px.box(filtered_df, x="employment_type", y="monthly_income", color="target_label",
                    title="Income by Employment Type")
st.plotly_chart(fig_income, use_container_width=True)

st.markdown("---")

# --------------------------
# 🔹 Insight 5: Delinquency Predictors
# --------------------------
st.subheader("🔹 Insight 5: Recent delinquency and risk grades are powerful predictors of default")
st.markdown("""
- Applicants with a recent delinquency flag and risk grade D/E show significantly higher rejection and fraud rates.
""")

fig_delinquency = px.histogram(filtered_df, x="delinquency_label", color="target_label",
                               barmode="group", title="Defaults vs Recent Delinquency")
st.plotly_chart(fig_delinquency, use_container_width=True)

fig_risk_delinq = px.histogram(filtered_df, x="risk_grade", color="delinquency_label",
                               barmode="group", title="Delinquency Flag by Risk Grade")
st.plotly_chart(fig_risk_delinq, use_container_width=True)

st.markdown("---")

# --------------------------
# 📊 Additional Analysis
# --------------------------
st.header("📊 Additional Analysis")

st.subheader("📍 Regional View")
fig_region = px.histogram(filtered_df, x="region", color="target_label", barmode="group",
                         title="Applications by Region")
st.plotly_chart(fig_region, use_container_width=True)

st.subheader("📈 Correlation Heatmap")

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
