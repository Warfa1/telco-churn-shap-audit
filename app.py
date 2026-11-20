import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Risk Predictor")
st.markdown("""
This app uses a **Logistic Regression** model to predict whether a customer is likely to churn.
It also uses **SHAP** to explain *why* the model made that prediction.
""")

@st.cache_resource
def load_model():
    return joblib.load('lr_model.pkl')

model = load_model()

st.sidebar.header("Customer Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 8000.0, 1000.0)
contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

if st.button("Predict Churn Risk"):
    
    input_data = {
        'gender': ['Female'], 'SeniorCitizen': [0], 'Partner': ['No'], 'Dependents': ['No'],
        'tenure': [tenure], 'PhoneService': ['Yes'], 'MultipleLines': ['No'],
        'InternetService': [internet], 'OnlineSecurity': ['No'], 'OnlineBackup': ['No'],
        'DeviceProtection': ['No'], 'TechSupport': ['No'], 'StreamingTV': ['No'],
        'StreamingMovies': ['No'], 'Contract': [contract], 'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'], 'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }
    
    input_df = pd.DataFrame(input_data)
    prob = model.predict_proba(input_df)[0, 1]
    
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Churn Probability", value=f"{prob*100:.1f}%")
    with col2:
        if prob > 0.5:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")
            
    st.subheader("Why did the model make this prediction?")
    
    preprocessor = model.named_steps['prep']
    classifier = model.named_steps['clf']
    
    input_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    
    explainer = shap.LinearExplainer(classifier, input_transformed)
    shap_values = explainer.shap_values(input_transformed)
    
       # Clean up feature names for a better display
    clean_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # max_display=6 shows only the top 6 most important features (thicker bars!)
    shap.bar_plot(shap_values[0], feature_names=clean_names, max_display=6, show=False)
    plt.tight_layout()
    st.pyplot(fig)
