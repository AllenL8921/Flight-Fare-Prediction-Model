import joblib
import pandas as pd

# Load the model and feature names
model = joblib.load("./model/flight_price_model.joblib")
feature_names = joblib.load("./model/model_features.joblib")


def predict_flight_price(input_data):
    """
    Predict flight price based on input features

    Parameters:
    input_data (dict): Dictionary containing flight details

    Returns:
    float: Predicted flight price
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all required features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction
