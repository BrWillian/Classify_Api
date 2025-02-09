from flask import Flask, jsonify, request, render_template, redirect
import numpy as np
import _pickle as pickle
import cv2

app = Flask(__name__)

svm = pickle.load(open("svm_model.pkl", "rb"))

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return render_template('upload.html')


@app.route('/predict', methods=["POST"])
def predict():

    requested_img = request.files['file']

    if requested_img and allowed_file(requested_img.filename):

        image = np.asarray(bytearray(requested_img.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
        image = image.flatten()
        image = image / np.mean(image)

        x = image.reshape(1, -1)

        prediction = svm.predict(x)[0]

        return jsonify({"Result": prediction})

    return redirect('/error')


@app.route('/error')
def error():
    return render_template('error.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)