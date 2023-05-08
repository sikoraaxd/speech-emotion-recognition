from flask import Flask, jsonify, request
import os
from scripts.predict import predict

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    data = request.get_json()
    filename = data['filename']
    b64data = data['data']
    with open(f'.src/{filename}', 'wb') as f:
        f.write(b64data)

    pred = predict()
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ['PORT']))