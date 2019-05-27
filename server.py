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

# return Index page
@app.route('/')
def index():
    
    return render_template('index.html')

# recognition digit
@app.route('/DigitRecognition', methods=['GET', 'POST'])
def ExecPy():
    retJson = {"predict_digit" :"Err", "detect_img" :"", "centering_img" :"", "prob" :{}}
    if request.method == 'POST':
        # request.body
        postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
        res =  cnn.predict(postImg) 
        
        if res is not None:
            retJson["predict_digit"] = str(np.argmax(res))

            for i, item in enumerate(res):
                retJson["prob"][i] = float(item*100)
        
            # save digits 
            postImg = Image.open(postImg)
            postImg.save("./predict_results/{}_{}.png".format(datetime.now().strftime('%m-%d_%H.%M.%S'),retJson["predict_digit"] ))

    return json.dumps(retJson)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    

