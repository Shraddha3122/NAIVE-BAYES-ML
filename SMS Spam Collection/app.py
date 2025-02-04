#Take the SMS spam collection dataset and analyze which messages are spam and
#which are ham by creating a spam filter using Multinomial Naïve Bayes Machine Learning Model.

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/NAIVE BAYES ML/SMS Spam Collection/SMSSpamCollection', sep='\t', names=['label', 'message'])

# Preprocess the data
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  
X = data['message']
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=44)

# Text vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the Multinomial Naïve Bayes 
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get input data from request
    input_data = request.json
    message = input_data['message']
    
    # Vectorize the input message
    message_vectorized = vectorizer.transform([message])
    
    # prediction
    prediction = model.predict(message_vectorized)
    
    #  result
    return jsonify({'is_spam': 'Yes' if prediction[0] == 1 else 'No'})

if __name__ == '__main__':
    app.run(debug=True)