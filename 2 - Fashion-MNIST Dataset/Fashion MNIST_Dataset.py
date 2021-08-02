# Baseline CNN model for Fashion MNIST Dataset
from numpy import mean, std
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.models import Sequential
from keras.datasets import fashion_mnist
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

"""The function loading the train and test dataset. The images are all pre-segmented (e.g. each image contains a
   single item of clothing), that the images all have the same square size of 28 × 28 pixels, and that the images 
   are grayscale. Therefore, we can load the images and reshape the data arrays to have a single color channel."""
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

"""The function prepares the pixel data. The pixel values for each image in the dataset are unsigned 
   integers in the range between [0,255]. The function normalizes the pixel values of grayscale images, 
   e.g. rescale them to the range [0,1]. This involves first converting the data type from unsigned 
   integers to floats, then dividing the pixel values by the maximum value."""
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm /= 255.0
    test_norm /= 255.0
    # return normalized images
    return train_norm, test_norm

"""The function defines a CNN (convolutional neural network) model."""
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')
    return model

"""The function evaluate a model using k-folds cross validation. The test set for each fold will be used to
   evaluate the model both during each epoch of the training run, so we can later create learning curves, 
   and at the end of the run, so we can estimate the performance of the model. As such, we will keep track 
   of the resulting history from each run, as well as the classification accuracy of the fold."""
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):    
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY = dataX[train_ix], dataY[train_ix],
        testX, testY = dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0, validation_data=(testX, testY))
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # append scores
        scores.append(acc)        
        histories.append(history)  

    return scores, histories

"""The function plots diagnostic learning curves. Once the model has been evaluated, we present the results. There
   are two key aspects to present: the diagnostics of the learning behavior of the model during training and the 
   estimation of the model performance. First, the diagnostics involve creating a line plot showing model 
   performance on the train and test set during each fold of the k-fold cross-validation. These plots are valuable 
   for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset."""
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.legend(['training set', 'test set'])
        pyplot.xlabel('number of k-Fold')
        pyplot.ylabel('cross entropy loss')
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[0].history['loss'], color='blue', label='train')
        pyplot.plot(histories[0].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.legend(['training set', 'test set'])
        pyplot.xlabel('number of k-Fold')
        pyplot.ylabel('accuracy')
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[0].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[0].history['val_accuracy'], color='orange', label='test')

    pyplot.tight_layout()
    pyplot.savefig('Fashion MNIST.png', dpi=900, transparent=True)
    pyplot.show()
 
"""The function summarizes model performance. The classification accuracy scores collected during each fold can be 
   summarized by calculating the mean and standard deviation. This provides an estimate of the average expected 
   performance of the model trained on this dataset, with an estimate of the average variance in the mean."""
def summarize_performance(scores):
    # print summary
    print('\nAccuracy: mean=%.3f, std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))

"""The function runs the test harness for evaluation a model. This involves calling all of the defined functions."""
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)

def main():
    # entry point, run the test harness
    run_test_harness()

if __name__ == '__main__':
    main()