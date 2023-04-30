Dog Cat CNN Classifier
This is a convolutional neural network (CNN) classifier designed to classify images of dogs and cats. The model is built using Python and Keras, and trained on a dataset of 25,000 images of dogs and cats.

Requirements
To run this classifier, you will need:

Python 3.x
TensorFlow 2.x or higher
Keras 2.x or higher
NumPy
Matplotlib
Jupyter Notebook (optional)
Installation
To install the required libraries, you can use the following command:
pip install tensorflow keras numpy matplotlib jupyter
Usage
To use the classifier, you can follow these steps:

Clone or download the repository.
Open the Jupyter Notebook file dog_cat_cnn_classifier.ipynb.
Follow the instructions in the notebook to train the model and test it on new images.
Alternatively, you can use the dog_cat_cnn_classifier.py file to train and test the model in a Python environment.

Dataset
The dataset used to train the model is the Kaggle Cats and Dogs Dataset, which consists of 25,000 images of cats and dogs. The dataset can be downloaded from here.

Model Architecture
The model architecture used in this classifier consists of four convolutional layers, followed by two fully connected (dense) layers. The model is trained using binary crossentropy loss and the Adam optimizer.

Results
The trained model achieves an accuracy of around 80% on the test set. However, the accuracy may vary depending on the specific images used for testing.

