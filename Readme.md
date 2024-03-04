# Model Summary

This Model is using the Injury prediction dataset which is accessible using the URL below 
https://www.kaggle.com/datasets/mrsimple07/injury-prediction-dataset

## Usage

You can use it easily by joblib library like below 
```
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

```

## System

Is this a standalone model or part of a system? What are the input requirements? What are the downstream dependencies when using the model outputs?

## Implementation requirements

The Injury_prediction.py script and the logistic regression model it contains can function both as a standalone model and as part of a larger system, depending on the use case and the context in which it's deployed. Here's a breakdown of its characteristics in both scenarios:

#### Standalone Model
Input Requirements: For the standalone model, the input requirements are specific and structured. It requires input in the form of a list or array that contains player information, specifically:
- Player_Age (numerical)
- Player_Weight (numerical)
- Player_Height (numerical)
- Previous_Injuries (numerical, e.g., count)
- Training_Intensity (numerical, continuous or discrete)
- Recovery_Time (numerical, e.g., days)
**Usage:** As a standalone tool, it could be used by coaches, sports scientists, or medical professionals to input player data manually and receive predictions on injury likelihood. This can aid in decision-making regarding training adjustments, recovery plans, or further medical assessments.
Outputs: The model outputs a binary prediction (‘High’ or ‘Low’ likelihood of injury) based on the input features.
#### Part of a Larger System
**Integration:** When integrated into a larger system, the model could automatically receive input data from a database or a data collection system that tracks player metrics over time. It could be part of a sports analytics platform, health monitoring system, or an athlete management system.
Input Requirements: In this scenario, the system would need to ensure that the data fed into the model matches the expected structure and format. Automated data pipelines might be required to preprocess data (e.g., normalization, filling missing values) before prediction.
#### Downstream Dependencies:
**Decision Support Systems:** The model's predictions could be used by other components within a system to provide recommendations or alerts regarding player management, training load adjustments, or preventative care.
**Monitoring and Alerting Systems:** Predictions indicating a high likelihood of injury could trigger alerts for further assessment or intervention.
**Data Analytics and Reporting Tools:** Model outputs could feed into analytics platforms for further analysis, tracking injury risk trends over time, or evaluating the effectiveness of training programs.
**Feedback Loop:** In a comprehensive system, model predictions could also be used to refine and improve the model over time. For instance, actual injury occurrences could be compared against model predictions to identify areas for model retraining or refinement.
#### Considerations for Integration
**Data Privacy and Security:** Especially relevant when dealing with health-related data, ensuring that the system complies with relevant data protection regulations (like GDPR or HIPAA in the healthcare sector).
**Scalability:** The ability to handle increased data volumes or to serve a larger number of users without degradation in performance.
**Maintenance and Updating:** The model might require periodic retraining to maintain its accuracy as new data becomes available or as player conditions and sports science evolve.
