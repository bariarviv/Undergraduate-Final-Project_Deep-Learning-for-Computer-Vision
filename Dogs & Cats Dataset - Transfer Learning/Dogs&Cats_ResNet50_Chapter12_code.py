# ResNet50 model used for Transfer Learning on the Dogs & Cats Dataset
import sys
from matplotlib import pyplot
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator

"""The function defines a CNN (convolutional neural network) model."""
def define_model():
	# load model
	model = ResNet50(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat = Flatten()(model.layers[-1].output)
	classifier = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat)
	output = Dense(1, activation='sigmoid')(classifier)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
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
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
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
	db = 'dataset_dogs_vs_cats/'
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = datagen.flow_from_directory(db + 'train/', class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory(db + 'test/', class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
								  validation_steps=len(test_it), epochs=10, verbose=1)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
	# save model
	model.save('dogs_vs_cats_vgg16.h5')

def main():
	# entry point, run the test harness
	run_test_harness()

if __name__ == '__main__':
	main()