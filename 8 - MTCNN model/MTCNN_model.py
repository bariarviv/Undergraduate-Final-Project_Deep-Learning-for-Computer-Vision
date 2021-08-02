# Face Detection with MTCNN model
import os
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Circle, Rectangle

"""The function load images from received folder."""
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path_image = os.path.join(folder, filename)
        img = cv2.imread(path_image)

        if img is not None:
            images.append(path_image)
            
    return images

"""The function draw an image with detected objects."""
def draw_image_with_boxes(filename, result_list):
    # load and plot the image
    data = pyplot.imread(filename)
    pyplot.axis('off')
    pyplot.imshow(data)
    # get the context for drawing boxes
    context = pyplot.gca()
    
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        context.add_patch(rect)
        
        # draw the dots
        for _, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            context.add_patch(dot)

    filename = filename[:filename.rindex('.')]
    # save the plot
    pyplot.savefig(filename + 'faces.png', dpi=900, transparent=True)
    # show the plot
    pyplot.show()
    
"""The function detect faces in the image with MTCNN model."""
def detect(filename):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    draw_image_with_boxes(filename, faces)

def main():
    folder = 'data'
    # define filenames
    filenames = load_images_from_folder(folder)
    # detect faces in the image with MTCNN
    for i in range(len(filenames)):
        detect(filenames[i])

if __name__ == '__main__':
    main()