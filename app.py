from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Airbnb Price Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert data to DataFrame
    df = pd.DataFrame(data, index=[0])

    # Preprocess the data (similar to what you did before training the model)
    # Drop unnecessary columns, encode categorical variables, etc.

    # Make predictions
    prediction = model.predict(df)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
