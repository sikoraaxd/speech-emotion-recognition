from flask import Flask, jsonify, request
import os
from scripts.predict import predict
import base64
from collections import Counter

app = Flask(__name__)

@app.route('/', methods=['GET'])
def getapi():
    return 'POST METHOD WITH JSON {"filename": "example.wav", "data": "base64 string"}'

@app.route('/', methods=['POST'])
def postapi():
    data = request.get_json()
    filename = data['filename']
    b64data = data['data']
    if 'src' not in os.listdir():
        os.mkdir('src')
    data = base64.b64decode(b64data.encode('utf-8'))
    with open(f'./src/{filename}', 'wb') as f:
        f.write(data)

    predictions = []
    for _ in range(20):
        _, pred = predict()
        predictions.append(pred)
    
    predictions = list(Counter(predictions).keys())
    
    os.remove(f'./src/{filename}')
    return jsonify({"filename": filename, "emotions": predictions})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ["PORT"]))