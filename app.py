"""
app.py — Streamlit frontend for the Iris ML API.
Sends user input to the FastAPI /predict endpoint and displays results.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

CLASS_EMOJI = {
    "setosa":     "🌸",
    "versicolor": "🌺",
    "virginica":  "🌼",
}

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the four measurements below and click **Predict**.")

# ── Input sliders ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal width  (cm)", 2.0, 4.5, 3.5, 0.1)

with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("Petal width  (cm)", 0.1, 2.5, 0.2, 0.1)

# ── Predict button ───────────────────────────────────────────────
if st.button("Predict", type="primary"):
    payload = {"features": [sepal_length, sepal_width, petal_length, petal_width]}

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        name  = result["predicted_class_name"]
        emoji = CLASS_EMOJI.get(name, "🌿")
        probs = result["probabilities"]

        st.success(f"**Predicted species: {emoji} {name.capitalize()}**")

        st.subheader("Class probabilities")
        for cls, prob in probs.items():
            st.progress(prob, text=f"{cls}: {prob:.2%}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure `uvicorn main:app --reload` is running.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ── Sidebar info ─────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("This app uses a **Random Forest** model trained on the classic Iris dataset.")
    st.write("**API:** FastAPI + Uvicorn")
    st.write("**Tracking:** MLflow")
    st.markdown("---")
    st.write("Make sure both services are running:")
    st.code("uvicorn main:app --reload\nstreamlit run app.py")
