## Digit Recognition WebApp using TensorFlow and Flask
**Demo Site : http://naikon.server-on.net/DigitRecognition/**

![Digit Recognition](./demo.gif)


### Specific

#### Neural Network

- using MNIST digit data
- 99.3% of accuracy
- 2 CNN layers, Adam Optimizer, Dropout
- centering input digit for better recognition

#### Web Application

- using Flask web framework (But the demo site uses Nginx + uWSGI)
- using d3.js for drawing bar graph

### Requirement
- Python 3.6
- TensorFlow 1.9
- Flask
- NumPy
- PIL(pillow)

### Usage

- #### Training Model
  `python3 train.py`

- #### Run WebApp
  `python3 server.py` then access to localhost:5000
  
