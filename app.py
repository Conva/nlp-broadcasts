from flask import Flask, request
from src.predict import predict_event_categories

app = Flask(__name__)



@app.route("/predict/events/category/<name>", methods = ['POST'])
def predict(name : str):
    phrase = request.data.decode('UTF-8')
    return predict_event_categories(name, phrase)
