from flask import Flask, render_template, request
from ConvNeural import ConvNeuralNet
from io import BytesIO
import os, base64
import numpy as np
from PIL import Image

app = Flask(__name__)
cnn = ConvNeuralNet()

# index にアクセスしたときの処理
@app.route('/')
def index():
    
    return render_template('index.html')

# /DigitRecognitionにPOSTされた画像で数字認識
@app.route('/DigitRecognition', methods=['GET', 'POST'])
def ExecPy():

    retStr = "Err"
    if request.method == 'POST':

        res =  cnn.Predict(BytesIO(base64.urlsafe_b64decode(request.form['img'])))        
        if res is not None:
            resIdx = np.argmax(res)
            print("推測結果:{0} .. [{1:.2f}%]".format(resIdx, res[resIdx]*100))
            retStr = "推測結果:{0} .. [{1:.2f}%]".format(resIdx, res[resIdx]*100)

    return retStr


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

