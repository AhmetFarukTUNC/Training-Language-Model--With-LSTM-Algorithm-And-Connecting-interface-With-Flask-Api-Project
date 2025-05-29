from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./results/model.h5"
TOKENIZER_PATH = "./results/tokenizer2.json"
MAX_SEQUENCE_LENGTH = 100

# Modeli yükle
model = load_model(MODEL_PATH)

# Tokenizer'ı yükle
with open(TOKENIZER_PATH, "r", encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Metni işleyip modele ver
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    prediction = model.predict(padded)[0][0]

    # Tahmini etiketle
    label = 1 if prediction >= 0.087 else 0

    labels = ['Çocuk', 'Yetiskin']
    label_name = labels[label]
    print(prediction)

    print(label_name)

    return jsonify({
        'prediction_score': float(prediction),
        'label': label_name
    })

if __name__ == '__main__':
    app.run(debug=True)
