# Classification of the MNIST Dataset Handwritten Digits using Dense Neural Network
from matplotlib import pyplot
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.models import Sequential 
from tensorflow.keras.utils import to_categorical

"""The mathematical notation x is used to represent the data we're feeding into a model as input, while y 
   is used for the labeled output that  we’re training the model to predict. With this in mind, X_train 
   stores the MNIST digits we'll be training our model on. Executing X_train.shape yields the output 
   (60000, 28, 28). This shows us that, as expected, we have 60,000 images in our training dataset, each 
   of which is a 28×28 matrix of values. Running y_train.shape, we unsurprisingly discover we have 60,000 
   labels indicating what digit is contained in each of the 60,000 training images."""
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

pyplot.figure(figsize=(5,5))
for i in range(12):
    pyplot.subplot(3, 4, i + 1)
    pyplot.imshow(X_train[i], cmap='Greys')
    pyplot.axis('off')

pyplot.tight_layout()
pyplot.savefig('MNIST.png', transparent=True)
pyplot.show()

""""We'll flatten our 28x28-pixel images into 784-element arrays. We employ the reshape() method."""
X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')

"""We divide all of the values by 255 so that they range from 0 to 1."""
X_train /= 255
X_valid /= 255

"""There are 10 possible handwritten digits, so we set n_classes equal to 10. In the other two lines of
   code we use a convenient utility function to_categorical, which is provided within the Keras library 
   to transform both the training and the validation labels from integers into the one-hot format."""
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_valid = to_categorical(y_valid, n_classes)

"""The label zero would be represented by a lone 1 in the first position, one by a lone 1 in the second 
   position, and so on. array([0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.], dtype=float32)"""
"""We instantiate the simplest type of neural network model object, the Sequential type. In the second line, we use 
   the add() method of our model object to specify the attributes of our network’s hidden layer (64 sigmoid-type 
   artificial neurons in the general-purpose, fully connected arrangement defined by the Dense() method) as well 
   as the shape of our input layer (one-dimensional array of length 784). In the third and final line we use the 
   add() method again to specify the output layer and its parameters: 10 artificial neurons of the softmax variety, 
   corresponding to the 10 probabilities (one for each of the 10 possible digits) that the network will output 
   when fed a given handwritten image."""
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])

"""The critical aspects are:
   1. The fit() method of our model object enables us to train our artificial neural network with the 
      training images X_train as inputs and their associated labels y_train as the desired outputs.
   2. As the network trains, the fit() method also provides us with the option to evaluate the performance 
      of our network by passing our validation data X_valid and y_valid into the validation_data argument.
   3. With machine learning, and especially with deep learning, it is commonplace to train our model on 
      the same data multiple times. One pass through all of our training data (60,000 images in the current 
      case) is called one epoch of training. By setting the epochs parameter to 200, we cycle through all
      60,000 training images 200 separate times.
   4. By setting verbose to 1, the model.fit() method will provide us with plenty of feedback as we train."""
history = model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=0, validation_data=(X_valid, y_valid))

# evaluate model
_, acc = model.evaluate(X_valid, y_valid, verbose=0)
print('> %.3f' % (acc * 100.0))