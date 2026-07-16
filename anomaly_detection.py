import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from data_loader import preprocess_data

@st.cache_resource
def train_anomaly_detection(data):
    data = preprocess_data(data)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(data.drop("class", axis=1))
    return iso_forest

def anomaly_detection(data):
    st.header("Anomaly Detection")
    st.write("Identify unusual or fraudulent loan applications.")

    # Load the anomaly detection model
    iso_forest = train_anomaly_detection(data)

    # Detect anomalies
    anomalies = data[iso_forest.predict(preprocess_data(data).drop("class", axis=1)) == -1]
    st.write("### Detected Anomalies")
    st.write(f"{len(anomalies)} unusual loan applications detected. These applications may require further review.")
    st.write(anomalies)