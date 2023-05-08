from flask import Flask, jsonify, request
import os
from scripts.predict import predict
import base64

app = Flask(__name__)

@app.route('/', methods=['GET'])
def getapi():
    return 'POST METHOD WITH JSON {"filename": "example.wav", "data": "base64 string"}'

@app.route('/', methods=['POST'])
def postapi():
    data = request.get_json()
    filename = data['filename']
    b64data = data['data']
    data = base64.b64decode(b64data.encode('utf-8'))
    with open(f'./src/{filename}', 'wb') as f:
        f.write(data)

    pred = predict()

    os.remove(f'./src/{filename}')
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ["PORT"]))