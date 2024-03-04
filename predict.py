import joblib
import pandas as pd

def predict_injury_likelihood(player_info):
    """
    Predicts the likelihood of injury given player information.
    
    Parameters:
    - player_info: A list or array-like object containing player information
                    in the order: [Player_Age, Player_Weight, Player_Height, Previous_Injuries, Training_Intensity, Recovery_Time]
    
    Returns:
    - injury_likelihood: A string indicating the likelihood of injury ('High' or 'Low').
    """
    # Load the saved model and scaler
    model = joblib.load('./injury_model.joblib')
    scaler = joblib.load('./scaler.joblib')
    
    # Convert player_info to a DataFrame with appropriate column names
    columns = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity', 'Recovery_Time']
    player_info_df = pd.DataFrame([player_info], columns=columns)
    
    # Scale the input features
    player_info_scaled = scaler.transform(player_info_df)
    
    # Predict the likelihood of injury
    prediction = model.predict(player_info_scaled)[0]
    
    # Interpret the prediction
    injury_likelihood = 'High' if prediction == 1 else 'Low'
    print(f"Prediction: {prediction}")
    
    return injury_likelihood

# Example usage
if __name__ == "__main__":
    # Example player information
    player_info = [55, 167, 183, 10, 2, 20]  # Example values for player information
    prediction = predict_injury_likelihood(player_info)
    print(f"Predicted Likelihood of Injury: {prediction}")
