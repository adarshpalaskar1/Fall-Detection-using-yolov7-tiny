# Fall-Detection-using-yolov7-tiny

The project aims to use TensorFlow Lite (TFLite) and YOLOv7tiny object detection model to create a real-time fall detection system using a Raspberry Pi. The TFLite framework will be used to deploy the YOLOv7tiny model onto the Raspberry Pi. The TFLite runtime library will be used to perform inference on the model.

The YOLOv7tiny model is a lightweight version of the popular YOLO (You Only Look Once) object detection algorithm. It is optimized for speed and can detect objects with high accuracy while running on low-resource devices like the Raspberry Pi.

The system will use a camera connected to the Raspberry Pi to capture real-time video feed. The captured video will be processed by the YOLOv7tiny model running on the Raspberry Pi using TFLite. The model will be trained to detect the human body and predict the probability of a fall.

If the model detects a fall, it will trigger an alarm or send an alert to a designated person or system. The system can also be configured to record the video feed before and after the fall for further analysis.The project aims to use TensorFlow Lite (TFLite) and YOLOv7tiny object detection model to create a real-time fall detection system using a Raspberry Pi. The TFLite framework will be used to deploy the YOLOv7tiny model onto the Raspberry Pi. The TFLite runtime library will be used to perform inference on the model.

The YOLOv7tiny model is a lightweight version of the popular YOLO (You Only Look Once) object detection algorithm. It is optimized for speed and can detect objects with high accuracy while running on low-resource devices like the Raspberry Pi.

The system will use a camera connected to the Raspberry Pi to capture real-time video feed. The captured video will be processed by the YOLOv7tiny model running on the Raspberry Pi using TFLite. The model will be trained to detect the human body and we use a simple logic to detect if a person is falling.

If the model detects a fall, it will trigger an alarm or send an alert to a designated person or system. The system has been configured to display the video feed and issue a warning on the screen.




## How to Run the Model
If running on a when running on a raspberry pi follow the given instructions to run the model.
Using pip install required files. First cd to given directory and use. 
```http
  pip install -r requirements.txt
```
Then open the tflitecode.py file and using build option in toolbar, first compile and then execute the file. 
