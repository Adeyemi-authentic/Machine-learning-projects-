import streamlit as st
import joblib
import numpy as np
import glob
import pickle

st.set_page_config(layout="wide")


# Function to predict email spams
def email_classifier(features):
    # Load the trained machine learning model
    model = joblib.load('email-spam.pkl')
    # Convert input features to a numpy array and reshape it
    features_array = np.array(features).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features_array)
    # Return prediction
    return prediction[0]


def main():
    # Add header
    st.title('Email classifier ')
    st.write('Welcome to our email classifier app.')
    st.write("this model classifies emails into 2 classes spam, or not spam")

    # Add instructions
    st.write('Please enter an email')
    Message = st.text_input("Message", "Type here")

    # Predict button
    if st.button('Predict'):
        # Gather input features
        features = [Message]
        # Predict email
        prediction = email_classifier(features)
        if prediction == 1:
            st.success('Prediction: email- Spam')
        else:
            st.success('Prediction: No spam - Not spam')


if __name__ == '__main__':
    main()

