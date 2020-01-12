# Handwritten Digits Classifier
A sample Python scripts that show how to train the TensorFlow model to classify handwritten digits from the MNIST database

## Usage
To train the model run Training.py. The script will start the training, and then save the resulting model to trained_model (in tf format). 

Eventually, you can skip the training and run Inference.py. This will load the trained model (from the attached trained_model), and recognize handwritten digit from the selected test image. You can change that image by updating testImageIndex variable:
```python
testImageIndex = 100 # or whatever between 0 and 9999. Note that there are 10,000 images in the test dataset
```
## Screenshots
![Figure](/images/Model-structure.png)
Figure 1. Structure of the model

![Figure](/images/Model-training.png)
Figure 2. Training the model

![Figure](/images/Recognition.png)

Figure 3. Recognition result
