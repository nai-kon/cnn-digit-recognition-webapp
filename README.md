## Digit Recognition WebApp by CNN using TensorFlow and Flask
A Flask WebApp for Handwritten digit recognition using Convolution Neural Network.      
**Demo Site : http://naikon.server-on.net/DigitRecognition/**

![Digit Recognition](./demo.gif)


### Specific

#### Neural Network
  
- Two Convolution - MaxPooling layer
- Softmax loss 
- Adam Optimizer
- Dropout
- Automatic Centering the Input Digit
- 99.3% of accuracy at MNIST test images


#### Web Application

- Flask for backbone (The Demo Site consists of Nginx + uWSGI)
- d3.js for bar graph

### Requirement
- Python 3.6
- TensorFlow 1.9
- Flask
- NumPy
- PIL(pillow)

### Usage

- #### Training Model
  ```
  - python3 train.py  
  ```
- #### Run WebApp
  ```
  - python3 server.py
    ->access to localhost:5000
  ```
  