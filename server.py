from io import BytesIO
from flask import Flask, render_template, request
from network import ConvNeuralNet
from train import MODEL_PATH
import base64
import json
import numpy as np
from datetime import datetime
import tensorflow as tf
from PIL import Image, ImageChops


app = Flask(__name__)
global_sess = None
model = ConvNeuralNet()

# return Index page
@app.route('/')
def index():
    return render_template('index.html')


# Recognition POST
@app.route('/DigitRecognition', methods=['POST'])
def ExecPy():
    retJson = {"predict_digit": "Err", "detect_img": "", "centering_img": "", "prob": {}}

    # request.body
    postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
    res = predict(postImg)

    if res is not None:
        retJson["predict_digit"] = str(np.argmax(res))

        for i, prob in enumerate(res):
            retJson["prob"][i] = prob * 100

        # save digits
        Image.open(postImg).save("./predict_results/{}_{}.png".format(datetime.now().strftime('%m-%d_%H.%M.%S'), retJson["predict_digit"]))

    return json.dumps(retJson)


# centering input digit
def __centering_img(img):

    w, h = img.size[:2]
    left, top, right, bottom = w, h, -1, -1
    imgpix = img.getdata()
    
    for y in range(h):
        yoffset = y * w
        for x in range(w):
            if imgpix[yoffset + x] < 255:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    shiftX = (left + (right - left) // 2) - w // 2
    shiftY = (top + (bottom - top) // 2) - h // 2

    return ImageChops.offset(img, -shiftX, -shiftY)


# predict digit
def predict(imgpath):
    try:
        img = Image.open(imgpath).convert('L')

    except IOError:
        print("image not found")
        return None

    # centering input digit
    img = __centering_img(img)

    img.thumbnail((28, 28))  # resize to 28x28
    img = np.array(img, dtype=np.float32)
    img = 1 - np.array(img / 255)  # invert and normalize
    img = img.reshape(1, 784)

    # predict
    res = global_sess.run(model.y_conv, feed_dict={model.x: img, model.y_: [[0.0] * 10], model.keep_prob: 1.0})[0]
    return res


if __name__ == "__main__":

    if not tf.train.checkpoint_exists(MODEL_PATH):
        print("no model to load")
        exit(1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
        global_sess = sess

        app.run(debug=True, host='0.0.0.0')
