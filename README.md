# Digit Recognition WebApp, PyTorch, Flask

**Demo Site : https://digit-recog-torch.uc.r.appspot.com/**

![Digit Recognition](./demo.gif)


### Model
- handwritten digit recognition using MNIST dataset
- two CNN layers, Batch Normalization, Adam Optimizer
- centering input digit for better recognition
- 99.3% of accuracy at validation

### Web Application
- Flask for web framework
- d3.js for drawing bar graph

## Requirements
- `pip install -r requirements.txt`
- Tested on Python 3.11.6, pytorch 2.4.1, CUDA 12.1

## Usage

- #### Run WebApp
  - `python3 server.py` (then access to `localhost:5000`)
  
- #### Training Model
  - Training on CPU: `python3 train.py`
  - Training on GPU: `python3 train.py --use_gpu` (enabled when gpu and cuda is available)
