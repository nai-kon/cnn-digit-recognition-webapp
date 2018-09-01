from io import BytesIO
from flask import Flask, render_template, request
from ConvNeural import ConvNeuralNet
import base64, json
import numpy as np
from PIL import Image
from datetime import datetime

app = Flask(__name__)
cnn = ConvNeuralNet()
cnn.loadmodel()

# index にアクセスしたときの処理
@app.route('/')
def index():
    
    return render_template('index.html')

# /DigitRecognitionにPOSTされた画像で数字認識
@app.route('/DigitRecognition', methods=['GET', 'POST'])
def ExecPy():

    retJson = {"predict_digit" : "Err", "detect_img" :"", "centering_img" : "", "prob" : {}}
    if request.method == 'POST':
        # request.body # IncompleteRead防止
        postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
        res =  cnn.predict(postImg) 
        print(res)
        
        if res is not None:

            retJson["predict_digit"] = str(np.argmax(res))
            #retJson["detect_img"] = imgtob64str(detect_img)
            #retJson["centering_img"] = imgtob64str(center_img)

            for i, item in enumerate(res):
                retJson["prob"][i] = float(item*100)
        
            # 入力数字と推測結果を保存
            postImg = Image.open(postImg)
            postImg.save("./predict_results/{}_{}.png".format(datetime.now().strftime('%m-%d_%H.%M.%S'),retJson["predict_digit"] ))

    return json.dumps(retJson)

def imgtob64str(img):
    imbuf = BytesIO()
    img.save(imbuf, "PNG")
    return str(base64.b64encode(imbuf.getvalue())).lstrip("b'").rstrip("'")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    

