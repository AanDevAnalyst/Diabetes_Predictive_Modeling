import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="🩺 Diabetes Status Predictor", layout="wide")

st.markdown(
    """
    <style>
        .big-font {font-size:22px !important; font-weight:600; color:#2E86C1;}
        .result-box {padding:10px; border-radius:10px; text-align:center; margin:10px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Prediction helper
# -------------------------------
def predict_diabetes_disease(df, models_name):
    predictions = []
    for model_name in models_name:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(df)
        predictions.append(prediction)
    return predictions


# -------------------------------
# Main Tabs
# -------------------------------
st.title("🩺 Diabetes Status Predictor")
tab1, tab2, tab3 = st.tabs(['🔮 Predict', '📦 Bulk Predict', '📊 Model Information'])

# -------------------------------
# Tab 1: Predict
# -------------------------------
with tab1:
    st.markdown('<p class="big-font">Enter Patient Information</p>', unsafe_allow_html=True)

    with st.sidebar:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=40, max_value=200, value=100)
        skin_thickness = st.number_input("Skin Thickness", min_value=7, max_value=99, value=20)
        insulin = st.number_input("Insulin Level", min_value=14, max_value=900, value=80)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=15.0, max_value=70.0, value=25.0, step=0.1)
        age = st.number_input("Age", min_value=18, max_value=90, value=30)

    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'Skin_Thickness': [skin_thickness],
        'Insulin': [insulin],
        'Body Mass Index': [bmi],
        'Age': [age]
    })

    algorithms = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Search']
    models_name = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'Gridrf.pkl']

    if st.button("🚀 Predict"):
        results = predict_diabetes_disease(input_data, models_name)

        for i, algo in enumerate(algorithms):
            if results[i][0] == 0:
                st.success(f"✅ {algo}: Patient is **Not Diabetic**")
            else:
                st.error(f"⚠️ {algo}: Patient is **Diabetic**")

# -------------------------------
# Tab 2: Bulk Predict
# -------------------------------
with tab2:
    st.markdown('<p class="big-font">Upload Patient Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        expected_columns = ['Pregnancies', 'Glucose', 'Skin_Thickness', 'Insulin', 'Body Mass Index', 'Age']

        if set(expected_columns).issubset(input_data.columns):
            for i, model_file in enumerate(models_name):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

                col_name = f"Prediction_{algorithms[i].replace(' ', '_')}"
                input_data[col_name] = model.predict(input_data[expected_columns])

            st.subheader("✅ Predictions Generated")
            st.dataframe(input_data, use_container_width=True)

            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=input_data.to_csv(index=False),
                file_name="Predictions.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Uploaded file does not contain required columns.")

# -------------------------------
# Tab 3: Model Information
# -------------------------------
with tab3:
    st.markdown('<p class="big-font">Model Accuracy Comparison</p>', unsafe_allow_html=True)

    data = {
        'Decision Tree': 70.78,
        'Logistic Regression': 70.78,
        'Random Forest': 70.13,
        'Support Vector Machine': 72.73,
    }
    df = pd.DataFrame(list(data.items()), columns=["Model", "Accuracy"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Model", y="Accuracy", data=df, palette="cool", ax=ax)

    for index, value in enumerate(df['Accuracy']):
        ax.text(index, value + 1, f"{value:.2f}%", ha='center', fontsize=10, weight='bold')

    ax.set_ylim(0, 100)
    ax.set_title("Model Accuracy Comparison", fontsize=14, weight='bold')
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    st.pyplot(fig)
    st.dataframe(df, use_container_width=True)
