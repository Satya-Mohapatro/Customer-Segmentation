# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation using K-Means Clustering")

st.write("""
This app demonstrates **K-Means Clustering** for customer segmentation using the Mall Customers Dataset.
You can:
- Upload your own customer CSV for segmentation.
- Test individual customer input for instant segment prediction.
- Visualize clusters interactively.
""")

# ------------------------
# Define Features (Required for .transform)
# ------------------------
numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_features = ['Gender']

# ------------------------
# Load Pre-fitted Pipeline and KMeans
# ------------------------
pipeline = joblib.load('pipeline.pkl')
kmeans = joblib.load('kmeans.pkl')

# ------------------------
# File Upload or Default Data
# ------------------------
st.sidebar.header("Upload CSV for Batch Segmentation")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded successfully.")
else:
    df = pd.read_csv("Mall_Customers.csv")
    st.info("ℹ️ Using default Mall Customers Dataset.")

# ------------------------
# Transform Data and Predict Clusters
# ------------------------
X = df[categorical_features + numeric_features]
X_processed = pipeline.transform(X)
df['Cluster'] = kmeans.predict(X_processed)

# ------------------------
# Display Cluster Distribution
# ------------------------
st.subheader("📊 Cluster Distribution")
cluster_counts = df['Cluster'].value_counts().sort_index().reset_index()
cluster_counts.columns = ['Cluster', 'Count']
st.dataframe(cluster_counts)

# ------------------------
# Cluster Visualization
# ------------------------
st.subheader("🎨 Cluster Visualization")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='tab10',
    s=100,
    alpha=0.7,
    ax=ax
)

# Plot centroids
centroids_scaled = kmeans.cluster_centers_

# Manually inverse transform numeric features only
scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
centroids_numeric_scaled = centroids_scaled[:, :len(numeric_features)]
centroids_numeric_original = scaler.inverse_transform(centroids_numeric_scaled)


ax.scatter(
    centroids_numeric_original[:, 1],  # Annual Income
    centroids_numeric_original[:, 2],  # Spending Score
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)


ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title('Customer Segments with Cluster Centroids')
ax.legend()
st.pyplot(fig)

# ------------------------
# Single Customer Prediction Using st.form
# ------------------------
st.sidebar.header("Predict Segment for a New Customer")

with st.sidebar.form("customer_form"):
    gender_input = st.selectbox("Gender", options=['Male', 'Female'])
    age_input = st.number_input("Age", min_value=18, max_value=70, value=30)
    income_input = st.number_input("Annual Income (k$)", min_value=15, max_value=150, value=50)
    spending_input = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    submit_button = st.form_submit_button(label="Predict Segment")

if submit_button:
    input_df = pd.DataFrame({
        'Gender': [gender_input],
        'Age': [age_input],
        'Annual Income (k$)': [income_input],
        'Spending Score (1-100)': [spending_input]
    })

    input_processed = pipeline.transform(input_df)
    cluster_pred = kmeans.predict(input_processed)[0]
    st.sidebar.success(f"The predicted customer segment is: **Cluster {cluster_pred}**")

