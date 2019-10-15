from flask import Flask, jsonify, request, render_template
import numpy as np
import _pickle as pickle
import cv2

svm = pickle.load(open("svm_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('upload.html')


@app.route('/predict', methods=["POST"])
def predict():

    requested_img = request.files['file']

    image = np.asarray(bytearray(requested_img.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    image = image.flatten()
    image = image / np.mean(image)

    x = image.reshape(1, -1)

    prediction = svm.predict(x)[0]

    return jsonify({"Result": prediction})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)