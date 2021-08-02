# Baseline model with dropout and data augmentation on the CIFAR10 Dataset.
import sys
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

"""The function loading the train and test dataset."""
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

"""The function prepares the pixel data. The pixel values for each image in the dataset are unsigned integers in the range 
   between [0,255]. The function normalizes the pixel values, e.g. rescale them to the range [0,1]. This involves first 
   converting the data type from unsigned integers to floats, then dividing the pixel values by the maximum value."""
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

"""The function defines a CNN (convolutional neural network) model."""
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

"""The function plots diagnostic learning curves. Once the model has been evaluated, we present the 
   results. There are two key aspects to present: the diagnostics of the learning behavior of the 
   model during training and the estimation of the model performance. First, the diagnostics involve 
   creating a line plot showing model performance on the train and test sets. These plots are valuable 
   for getting an idea of whether a model is overfitting, underfitting, or has a good fit for the dataset. 
   The plot is saved to file, specifically a file with the same name as the script with a png extension."""
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.legend(['training set', 'test set'])
	pyplot.xlabel('number of epochs')
	pyplot.ylabel('cross entropy loss')
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.legend(['training set', 'test set'])
	pyplot.xlabel('number of epochs')
	pyplot.ylabel('accuracy')
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.tight_layout()
	pyplot.savefig(filename + '_plot.png', dpi=900, transparent=True)
	pyplot.close()

"""The function runs the test harness for evaluation a model. This involves calling all of the defined functions."""
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	# prepare iterator
	it_train = datagen.flow(trainX, trainY, batch_size=64)
	# fit model
	steps = int(trainX.shape[0] / 64)
	history = model.fit(it_train, steps_per_epoch=steps, epochs=400, verbose=0, validation_data=(testX, testY))
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('Accuracy: %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
	# save model
	model.save('final_model_cifar10.h5')

def main():
	# entry point, run the test harness
	run_test_harness()

if __name__ == '__main__':
	main()