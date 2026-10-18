import streamlit as st
import joblib
import pandas as pd


model = joblib.load("model/random_forest_model.pkl")

st.set_page_config(
    page_title="NASA Exoplanet Classifier",
    page_icon="🌌",
    layout="wide"
)
st.title("🌌 NASA Exoplanet Classifier")
st.write(
    "Predict whether a Kepler exoplanet candidate is "
    "CONFIRMED or FALSE POSITIVE using Machine Learning."
)

st.divider()
st.header("🔭 Enter Exoplanet Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    koi_period = st.number_input(
        "Orbital Period",
        min_value=0.0,
        value=10.0
    )

    koi_duration = st.number_input(
        "Transit Duration",
        min_value=0.0,
        value=3.0
    )

    koi_depth = st.number_input(
        "Transit Depth",
        min_value=0.0,
        value=500.0
    )

with col2:
    koi_prad = st.number_input(
        "Planet Radius",
        min_value=0.0,
        value=1.2
    )

    koi_teq = st.number_input(
        "Equilibrium Temperature",
        min_value=0.0,
        value=500.0
    )

    koi_insol = st.number_input(
        "Insolation Flux",
        min_value=0.0,
        value=1.0
    )

with col3:
    koi_model_snr = st.number_input(
        "Model SNR",
        min_value=0.0,
        value=20.0
    )

    koi_steff = st.number_input(
        "Star Temperature",
        min_value=0.0,
        value=5500.0
    )

    koi_srad = st.number_input(
        "Star Radius",
        min_value=0.0,
        value=1.0
    )

st.divider()


if st.button("🚀 Predict Exoplanet", use_container_width=True):

    input_data = pd.DataFrame([{
        "koi_period": koi_period,
        "koi_duration": koi_duration,
        "koi_depth": koi_depth,
        "koi_prad": koi_prad,
        "koi_teq": koi_teq,
        "koi_insol": koi_insol,
        "koi_model_snr": koi_model_snr,
        "koi_steff": koi_steff,
        "koi_srad": koi_srad
    }])


    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.subheader("🔬 Prediction Result")

    if prediction[0] == "CONFIRMED":
        st.success("✅ EXOPLANET CONFIRMED")
    else:
        st.error("❌ FALSE POSITIVE")

    st.subheader("📊 Prediction Confidence")

    for class_name, prob in zip(model.classes_, probability[0]):
        st.write(
            f"**{class_name}** : {prob * 100:.2f}%"
        )
        st.progress(float(prob))

        st.divider()

st.header("📈 Model Feature Importance")

importance = model.feature_importances_
features = [
    "Orbital Period",
    "Transit Duration",
    "Transit Depth",
    "Planet Radius",
    "Equilibrium Temperature",
    "Insolation Flux",
    "Model SNR",
    "Star Temperature",
    "Star Radius"
]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.divider()

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "93%")

with col2:
    st.metric("Confirmed F1-Score", "0.89")

with col3:
    st.metric("False Positive F1-Score", "0.94")

with col4:
    st.metric("Macro F1-Score", "0.92")