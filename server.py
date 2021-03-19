import json

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageChops, ImageOps
from torchvision import transforms

from model import Model
from train import SAVE_MODEL_PATH

app = Flask(__name__)
predict = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/DigitRecognition", methods=["POST"])
def predict_digit():
    img = Image.open(request.files["img"]).convert("L")

    # predict
    res_json = {"pred": "Err", "probs": []}
    if predict is not None:
        res = predict(img)
        res_json["pred"] = str(np.argmax(res))
        res_json["probs"] = [p * 100 for p in res]

    return json.dumps(res_json)


class Predict():
    def __init__(self):
        device = torch.device("cpu")
        self.model = Model().to(device)
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def _centering_img(self, img):
        w, h = img.size[:2]
        left, top, right, bottom = w, h, -1, -1
        imgpix = img.getdata()

        for y in range(h):
            yoffset = y * w
            for x in range(w):
                if imgpix[yoffset + x] > 0:
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)

        shiftX = (left + (right - left) // 2) - w // 2
        shiftY = (top + (bottom - top) // 2) - h // 2
        return ImageChops.offset(img, -shiftX, -shiftY)

    def __call__(self, img):
        img = ImageOps.invert(img)  # MNIST image is inverted
        img = self._centering_img(img)
        img = img.resize((28, 28), Image.BICUBIC)  # resize to 28x28
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)  # 1,1,28,28

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor)
            preds = preds.detach().numpy()[0]

        return preds


if __name__ == "__main__":
    import os
    assert os.path.exists(SAVE_MODEL_PATH), "no saved model"
    predict = Predict()

    app.run(host="0.0.0.0")
