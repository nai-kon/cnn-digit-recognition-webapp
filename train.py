from ConvNeural import ConvNeuralNet
import sys

if __name__ == '__main__':

    cnn = ConvNeuralNet()
    if cnn.isTrained():
        if input('Trained model is exist. Overwrite it? y/n ') == "n":
            sys.exit()

    # Training network
    cnn.train()
