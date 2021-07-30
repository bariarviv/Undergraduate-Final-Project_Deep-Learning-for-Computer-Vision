# YOLOv4 model

From the versions of the YOLO model developed over the years, for detecting objects in an image and in real-time, we have chosen to train the YOLOv4 model. In order to train a model on top of a new object, a dataset that contains images containing the object and a text file for each image needs to be created.

In this case, we used the Bing API of Microsoft in order to download images containing kangaroos. The dataset we built is comprised of 800 square 416×416-pixel color images. 160 of the images for the validation dataset, and the rest, 640, were used for the training dataset.  
Next, we tagged all the images in the dataset using the [CVAT site](http://www.cvat.org), that is, for each image we put a bounding box around the kangaroo in the image and tagged it as a kangaroo. After doing this for all the images, we exported them to text files corresponding to the YOLO model.   
It will create .txt-file for each image in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line: 
`<object-class> <x_center> <y_center> <width> <height>`

<p align="center">
  <img src="images\cvat.png" width="700" height="400">
</p>

Finally, we performed the model training using darknet and followed the instructions for execution given by [AlexeyAB in his GitHub repository](http://www.github.com/AlexeyAB/darknet). The steps include:
* **Downloading the pre-trained weights file.**
* **Downloading the model configuration file and making several changes according to the instructions.**
* **Creating the label file with the names of the objects, each in a new line.**
* **Creating the data file containing:** the number of classes, the paths of the directories containing the training set, the validation set, the label file, and the path of the backup directory. At the end of the training, the backup directory contains the weights of the trained model.
* **Start training by using the command line:** ./darknet detector train yolov4.data yolov4.cfg yolov4.conv.137 -map

We determined the number of batches as 4000. A model file is created and saved at the end of each 1000 batches in a backup directory. From these files, we chose to use the model which minimized the loss function, thus giving us the best performance, in this case, it was the last file. In order to evaluate the model performance, we showed the plot of the calculated mAP. The performance of a model for an object detection task is often evaluated using the mean average precision, or mAP, which is calculated across all of the images in a dataset. These AP scores can be collected across a dataset and the mean calculated to give an idea of how good the model is at detecting objects in a dataset. We are predicting bounding boxes so we can determine whether a bounding box prediction is good or not based on how well the predicted and actual bounding boxes overlap. The parameters required for calculating mAP:
* **Precision** refers to the percentage of the correctly predicted bounding boxes out of all bounding boxes predicted.
* **Recall** is the percentage of the correctly predicted bounding boxes out of all objects in the photo.

As we make more predictions, the recall percentage will increase, but precision will drop or become erratic as we start making false-positive predictions. 

## Requirements
~~~bash
pip install requests
pip install argparse 
pip install opencv-python
~~~

## Results

The figure below shows the mAP score and the loss function. We can notice the aspiration of the loss function (blue curve) to zero as the training progresses and on the other hand, the mAP score (red curve) increases up to 99%. It can be deduced from the graph that the model was well trained, but with a larger dataset, better results could have been obtained. In addition, the output images of the trained model, which include one or more kangaroo detection in the image, are displayed below. We can see that the model has done well on these examples, finding all of the kangaroos, even in the case where there are two or three in one image.

<p align="center">
  <img src="results\YOLO.png" width="500" height="600">
</p>

<table align="center">
  <tr>
    <td><img src="results\5.jpg"></td>
    <td><img src="results\2.jpg"></td>
  </tr>
  <tr>
    <td><img src="results\3.jpg"></td>
    <td><img src="results\1.jpg"></td>
  </tr>
  <tr>
    <td><img src="results\4.jpg"></td>
    <td><img src="results\7.jpg"></td>
  </tr>
  <tr>
    <td><img src="results\6.jpg"></td>
    <td><img src="results\8.jpg"></td>
  </tr>
</table>
