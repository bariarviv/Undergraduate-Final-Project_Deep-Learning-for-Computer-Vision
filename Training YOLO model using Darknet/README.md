# YOLOv4 model

From the versions of the YOLO model developed over the years, for detecting objects in an image and in real-time, we have chosen to train the YOLOv4 model. In order to train a model on top of a new object, a dataset that contains images containing the object and a text file for each image needs to be created.

In this case, we used the Bing API of Microsoft in order to download images containing kangaroos. The dataset we built is comprised of 800 square 416×416-pixel color images. 160 of the images for the validation dataset, and the rest, 640, were used for the training dataset.  
Next, we tagged all the images in the dataset using the [CVAT site](http://www.cvat.org), that is, for each image we put a bounding box around the kangaroo in the image and tagged it as a kangaroo. After doing this for all the images, we exported them to text files corresponding to the YOLO model.   
It will create .txt-file for each image in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line: 
`<object-class> <x_center> <y_center> <width> <height>`

Finally, we performed the model training using darknet and followed the instructions for execution given by [AlexeyAB in his GitHub repository](http://www.github.com/AlexeyAB/darknet). The steps include:
* **Downloading the pre-trained weights file.**
* **Downloading the model configuration file and making several changes according to the instructions.**
* **Creating the label file with the names of the objects, each in a new line.**
* **Creating the data file containing:** the number of classes, the paths of the directories containing the training set, the validation set, the label file, and the path of the backup directory. At the end of the training, the backup directory contains the weights of the trained model.
* **Start training by using the command line:** ./darknet detector train yolov4.data yolov4.cfg yolov4.conv.137 -map



## Requirements
~~~bash
pip install requests
pip install argparse 
pip install opencv-python
~~~

## Results
