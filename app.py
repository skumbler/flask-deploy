from ultralytics import YOLO
import torch
import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import io
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line to enable CORS (Cross-Origin Resource Sharing)

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS (Cross-Origin Resource Sharing)


def Predict(img):
    model = YOLO("D:/flask_app/best.pt")
    results = model(img, verbose=False)

    for result in results:
        classes_names = result.names[result.probs.top1]
        probs = result.probs.top1conf.item()

    return {"Class": classes_names, "Confidence": probs}


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file)

    result = Predict(img)

    return jsonify(result)


if __name__ == '__main__':
    app.run()
