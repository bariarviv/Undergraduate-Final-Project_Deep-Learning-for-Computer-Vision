# Mask R-CNN model: Example of inference with a pre-trained COCO model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

"""define 81 classes that the coco model knowns about."""
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',  'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
               'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

"""The configuration object define how the model might be used during training or interface."""
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     # specify the number of images per batch
     IMAGES_PER_GPU = 1
     # the number of classes to predict
     NUM_CLASSES = 1 + 80

def main():
    # define the model: the model must be defined via an instance of the MaskRCNN class
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
    # load coco model weights
    rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
    # load photograph
    img = load_img('elephant.jpg')
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    res = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    display_instances(img, res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])

if __name__ == '__main__':
    main()