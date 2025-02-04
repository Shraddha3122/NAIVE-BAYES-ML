#  We are given weather data related to outlook, temperature, humidity and windy.
#Analyze the data using Gaussian Naive Bayes classifier and predict whether cricket can be played
#or not based on given new data.

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/NAIVE BAYES ML/Cricket/cricket.csv')

# Preprocess the data
data = pd.get_dummies(data, columns=['OUTLOOK', 'TEMPERATURE', 'HUMIDITY', 'WINDY'], drop_first=True)

# Split the data into features and target variable
X = data.drop('PLAY CRICKET', axis=1)
y = data['PLAY CRICKET'].map({'Yes': 1, 'No': 0})

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
   
    # Get input data 
    input_data = request.json
    
    # Convert input data 
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input data 
    input_df = pd.get_dummies(input_df, columns=['OUTLOOK', 'TEMPERATURE', 'HUMIDITY', 'WINDY'], drop_first=True)
    
    # Align input data 
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Return the result
    return jsonify({'can_play_cricket': 'Yes' if prediction[0] == 1 else 'No'})

if __name__ == '__main__':
    app.run(debug=True)