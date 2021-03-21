from flask import Flask, request
from flask_cors import CORS
import pickle
from utils import process_text, make_predict

with open('RidgeClassifier.sav', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.sav', 'rb') as f:
    vectorizer = pickle.load(f)

predict = make_predict(model, vectorizer)

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=["POST"])
def predictroute():
    if request.method == "POST":
        text = request.get_json()['text']
        score = predict(text)
        return str(score)

if __name__ == '__main__':
    app.run(debug=True)