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

        for i, item in enumerate(res):
            retJson["prob"][i] = float(item * 100)

        # save digits
        Image.open(postImg).save("./predict_results/{}_{}.png".format(datetime.now().strftime('%m-%d_%H.%M.%S'), retJson["predict_digit"]))

    return json.dumps(retJson)


# centering input digit
def __centering_img(img):

    width, height = img.size[:2]
    left, top, right, bottom = width, height, -1, -1
    imgpix = img.getdata()

    for y in range(height):
        yoffset = y * width
        for x in range(width):
            if imgpix[yoffset + x] < 255:

                # do not use GetPixel and SetPixel, it is so slow.
                if x < left:
                    left = x
                if y < top:
                    top = y
                if x > right:
                    right = x
                if y > bottom:
                    bottom = y

    shiftX = (left + (right - left) // 2) - width // 2
    shiftY = (top + (bottom - top) // 2) - height // 2

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
    img = 1 - np.array(img / 255)  # normalize
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
