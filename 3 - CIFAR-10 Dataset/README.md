# CIFAR-10 Dataset

CIFAR is an acronym that stands for the Canadian Institute for Advanced Research and the CIFAR-10 dataset was developed along with the CIFAR-100 dataset by researchers at the CIFAR institute. The dataset is comprised of 60,000 32×32-pixel color photographs of objects from 10 classes, such as frogs, birds, cats, ships, etc. There are 50,000 examples in the training dataset and 10,000 in the test dataset. The class labels and their standard associated integer values are listed below:

0. airplane
1. automobile
2. bird
3. cat
4. deer
5. dog
6. frog
7. horse
8. ship
9. truck

These are very small images, much smaller than a typical photograph, as the dataset was intended for computer vision research. CIFAR-10 is a well-understood dataset and is widely used for benchmarking computer vision algorithms in the field of machine learning. The problem is solved. It is relatively straightforward to achieve 80% classification accuracy. Top performance on the problem is achieved by deep learning convolutional neural networks with a classification accuracy above 90% on the test dataset.

Listed below is the figure of the first nine images in the dataset. It can be challenging to see what exactly is represented in some of the images given the extremely low resolution. This low resolution is likely the cause of the limited performance that top-of-the-line algorithms are able to achieve on the dataset.

<p align="center">
  <img src="images\Fashin MNIST.png" width="600" height="560">
</p>

The baseline model was developed in order to estimate the performance of a model on the problem in general, in the example, 5-fold cross-validation was used. The value of k = 5 was chosen to provide a baseline for both repeated evaluation and to not be too large as to require a long running time. Each test set will be 20% of the training dataset, or about 12,000 examples, close to the size of the actual test set for this problem. The training dataset is shuffled prior to being split and the sample shuffling is performed each time so that any model evaluated will have the same train and test datasets in each fold.

## Requirements
~~~bash
pip install matplotlib 
pip install tensorflow 
pip install Keras 
pip install numpy
~~~

## Results

The diagnostics involve creating a line plot showing model performance on the train and test set during each fold of the k-fold cross-validation. These plots are valuable for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset. There are two subplots, one for loss and one for accuracy. Blue lines indicate model performance on the training dataset and orange lines indicate performance on the hold out test dataset. Next, the classification accuracy scores collected during each fold are summarized by calculating the mean and standard deviation. This provides an estimate of the average expected performance of the model trained on this dataset, with an estimate of the average variance in the mean.

<p align="center">
  <img src="results/Fashion MNIST1.png" width="550" height="450">
</p>

In this case, we can see that the model generally achieves a good fit, with train and test learning curves converging. There may be some signs of slight overfitting.
