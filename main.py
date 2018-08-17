import time
from ConvNeural import ConvNeuralNet
import numpy as np

if __name__ == '__main__':

    cnn = ConvNeuralNet()

    if cnn.isTrained():
        # 推測
        while True:
            imgpath = input('手書き数字画像のパス("exit"で終了)>  ')
            if imgpath == "exit": break
            if len(imgpath) == 0: continue

            res =  cnn.Predict(imgpath)        
            if res is not None:
                resIdx = np.argmax(res)
                print("推測結果:{0} .. [{1:.2f}%]".format(resIdx, res[resIdx]*100))
                # for i, ress in enumerate(res):
                #     print("[{0}]:{1:.2f}%".format(i, ress*100))            

    else:
        # 学習
        cnn.Training()
