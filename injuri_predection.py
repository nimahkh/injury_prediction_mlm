# Injury_prediction.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
df = pd.read_csv('./data/injury_data.csv')

# Prepare the data
X = df.drop('Likelihood_of_Injury', axis=1)
y = df['Likelihood_of_Injury']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler to disk for later use
joblib.dump(model, './injury_model.joblib')
joblib.dump(scaler, './scaler.joblib')

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
    
    # Scale the input features
    player_info_scaled = scaler.transform([player_info])
    
    # Predict the likelihood of injury
    prediction = model.predict(player_info_scaled)[0]
    
    # Interpret the prediction
    injury_likelihood = 'High' if prediction == 1 else 'Low'
    
    return injury_likelihood

# Example usage
if __name__ == "__main__":
    # Example player information
    player_info = [24, 70, 180, 1, 0.5, 4]  # Example values for player information
    prediction = predict_injury_likelihood(player_info)
    print(f"Predicted Likelihood of Injury: {prediction}")
