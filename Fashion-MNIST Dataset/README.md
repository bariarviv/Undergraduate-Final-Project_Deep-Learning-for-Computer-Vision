# Fashion MNIST Dataset

The Fashion-MNIST dataset is proposed as a more challenging replacement dataset for the MNIST dataset. It is a dataset comprised of 60,000 small square 28×28-pixel grayscale images of items of 10 types of clothing, such as shoes, t-shirts, dresses, etc.  

The mapping of all 0-9 integers to class labels is listed below:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

It is a more challenging classification problem than MNIST and top results are achieved by deep learning convolutional neural networks with a classification accuracy of about 90% to 95% on the hold out test dataset.

<p align="center">
  <img src="images\Fashin MNIST.png" width="650" height="600">
</p>

The baseline model was developed in order to estimate the performance of a model on the problem in general, in the example, 5-fold cross-validation was used. The value of k=5 was chosen to provide a baseline for both repeated evaluation and to not be too large as to require a long running time. Each test set will be 20% of the training dataset, or about 12,000 examples, close to the size of the actual test set for this problem. The training dataset is shuffled prior to being split and the sample shuffling is performed each time so that any model evaluated will have the same train and test datasets in each fold.

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
