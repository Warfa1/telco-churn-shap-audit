import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import urllib.request

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Risk Predictor")
st.markdown("""
This app uses a **Logistic Regression** model to predict whether a customer is likely to churn.
It also uses **SHAP** to explain *why* the model made that prediction.
""")

@st.cache_resource
def load_model_and_data():
    # 1. Load the trained model
    model = joblib.load('lr_model.pkl')
    
    # 2. Download the dataset to use as SHAP background data
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Take a small sample for background
    X_background = df.drop(columns=['Churn']).head(100)
    
    return model, X_background

model, background_data = load_model_and_data()

st.sidebar.header("Customer Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 8000.0, 1000.0)
contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

if st.button("Predict Churn Risk"):
    
    # Create the input dataframe with the selected values
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
    
    # Make the prediction
    prob = model.predict_proba(input_df)[0, 1]
    
    # Display the result
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
    
    # Prepare data for SHAP
    preprocessor = model.named_steps['prep']
    classifier = model.named_steps['clf']
    
    input_transformed = preprocessor.transform(input_df)
    bg_transformed = preprocessor.transform(background_data)
    
    # Clean up feature names for a better display
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [name.replace('num__', '').replace('cat__', '') for name in raw_names]
    
    # Calculate SHAP values using the background data
    explainer = shap.LinearExplainer(classifier, bg_transformed)
    shap_values = explainer.shap_values(input_transformed)
    
    # Plot the SHAP bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.bar_plot(shap_values[0], feature_names=clean_names, max_display=8, show=False)
    plt.tight_layout()
    st.pyplot(fig)
