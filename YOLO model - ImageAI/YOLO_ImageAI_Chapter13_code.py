# Using a trained YOLO model from the ImageAI library to perform object detection in the video
from imageai.Detection import VideoObjectDetection

def detection(input_path, output_path, yolo_model_path):
    """The function opens the input video and goes through each frame. Performs object recognition for
       each frame by using a YOLO model and writes the frame including the detection to the output video
       :param input_path:      input video path
       :type input_path:       string
       :param output_path:     output video path
       :type output_path:      string
       :param yolo_model_path: YOLO model path
       :type yolo_model_path:  string
       :return: None
    """
    detector = VideoObjectDetection()
    # this function sets the model type of the object detection instance you created to the YOLOv3 model
    detector.setModelTypeAsYOLOv3()
    # this function accepts a string that must be the path to the model file,
    # it must correspond to the model typeset for the object detection instance
    detector.setModelPath(yolo_model_path)
    # this function loads the model from the path given
    detector.loadModel()

    # the function performs object detection on a video file or video live-feed
    # after  the model has been loaded into the instance that was created
    detector.detectCustomObjectsFromVideo(input_file_path=input_path, output_file_path=output_path,
                                          frames_per_second=20, log_progress=True)

def main():
    yolo_model_path = 'yolov3.h5'
    input_path = 'input_video.mp4'
    output_path = 'output_video.avi'
    detection(input_path, output_path, yolo_model_path)

if __name__ == "__main__":
    main()