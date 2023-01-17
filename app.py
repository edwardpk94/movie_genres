"""
Model API file
"""

from flask import Flask, jsonify, request
import dill
import pandas as pd

from src.preprocessing import clean_text
from config.paths import GENRES_PATH

app = Flask(__name__)
model = dill.load(open('models/rf_pipeline.pkl', 'rb'))
genres = pd.read_csv(GENRES_PATH)['genre'].tolist()


@app.route('/predict',methods=['POST'])
def predict():
    """
    Model prediction method

    Expects "text" parameter in the request body
    """
    req_data = request.get_json()
    print(req_data)
    text = req_data['text']
    text = [clean_text(text)]
    pred = model.predict(text)
    pred = pred.flatten()
    pred_genres = [genres[i] for i in range(len(pred)) if pred[i] == 1]
    response = {
        'prediction': pred_genres
    }
    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    """
    Show API index page
    """
    return 'Movie Genre Prediction API'

if __name__ == "__main__":
    app.run(debug=True)