import streamlit as st
import pandas as pd
import joblib

# Set the title of the web app
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.write("This app predicts whether a passenger would have survived the Titanic disaster.")

# Load your trained model
# Use @st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        st.error("Please run `train_model.py` first to create the model file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model('titanic_model.joblib')

# Only show the prediction UI if the model loaded successfully
if model is not None:
    # --- Create Input Fields for User ---
    st.header("Passenger Details")

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # Pclass (Passenger Class)
        pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3],
                              help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
        
        # Sex
        sex_str = st.radio("Sex:", ["Male", "Female"])
        # Map string to number
        sex = 0 if sex_str == "Male" else 1

        # Age
        age = st.slider("Age:", 0, 80, 25, help="Passenger's age in years.")

    with col2:
        # SibSp (Siblings/Spouses Aboard)
        sibsp = st.number_input("Siblings/Spouses Aboard (SibSp):", 0, 8, 0)
        
        # Parch (Parents/Children Aboard)
        parch = st.number_input("Parents/Children Aboard (Parch):", 0, 6, 0)
        
        # Fare
        fare = st.number_input("Fare ($):", 0.0, 512.0, 10.0, format="%.2f")
    
    # Embarked (Port of Embarkation)
    embarked_str = st.selectbox("Port of Embarkation (Embarked):", 
                                ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
                                help="Port where the passenger boarded.")
    # Map string to number
    if embarked_str == "Southampton (S)":
        embarked = 0
    elif embarked_str == "Cherbourg (C)":
        embarked = 1
    else:
        embarked = 2

    # --- Prediction Logic ---
    if st.button("Predict Survival", type="primary"):
        # Create a DataFrame from the user's inputs
        # The order MUST match the 'features' list from train_model.py
        input_data = pd.DataFrame(
            [[pclass, sex, age, sibsp, parch, fare, embarked]],
            columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        )
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Display the result
            st.header("Prediction Result")
            if prediction == 1:
                st.success("🎉 This passenger likely SURVIVED! 🎉")
                st.write(f"Confidence: {prediction_proba[1]*100:.2f}%")
            else:
                st.error("💔 This passenger likely DID NOT SURVIVE. 💔")
                st.write(f"Confidence: {prediction_proba[0]*100:.2f}%")

            st.write("---")
            st.subheader("Input Data:")
            st.dataframe(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Model file `titanic_model.joblib` not found. Please add it to your repository.")
