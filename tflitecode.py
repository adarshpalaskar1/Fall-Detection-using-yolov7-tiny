# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import cv2
import tflite_runtime.interpreter as Interpreter
# construct the argument parse and parse the arguments

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [ "person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

fps = FPS().start()
prev_centroid = 0
centroid = 1000
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # predictions
    frame = cv2.resize(frame, (640,640))
    # load model
    # Load the TFLite model and allocate tensors.
    interpreter = Interpreter(model_path="yolov7_model.tflite")
    interpreter.allocate_tensors()
    im = frame.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (640, 640))
    im = im / 255.
    # Change data layout from HWC to CHW
    im = im.transpose((2, 0, 1))
    im = im[np.newaxis, ...].astype(np.float32)
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], im)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    ori_images = [frame.copy()]
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box = box.round().astype(np.int32).tolist()        
        cv2.rectangle(image,box[:2],box[2:],2)
        #Calculate the centroid of the bounding box
        centroid = (int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0))
        # Draw the detected people
        #Draw circles around the detected people
        cv2.circle(image, centroid, 5, (0, 0, 255), -1)
        if prev_centroid != 0:
            if centroid[1] < (prev_centroid * 0.8):
                cv2.putText(image, "Warning: Person has fallen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        prev_centroid = centroid[1]
    # # Convert boxes to array
    # boxes = np.array(boxes)
    # if boxes[0][1]==0:
    #     rects = [0,0,0,0]
    # else:
    #     rects = boxes[0]
    # # Detect centroids of the detected people
    # centroids = []
    # xA = int(rects[0])
    # yA = int(rects[1])
    # xB = int(rects[2])
    # yB = int(rects[3])
    # centroids.append((int((xA + xB) / 2.0), int((yA + yB) / 2.0)))
    # # Draw the detected people
    # # Draw rectangles around the detected people
    # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    # x = centroids[0][0]
    # y = centroids[0][1]
    # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    # loop over the detections
    # Calculate the centroid of the bounding box
    # print(centroids)

    # if len(centroids)!=0:
    #     centroid = centroids[0][1]
    #     # print(centroid)
    # # If the centroid is 10% of the previous centroid then give a warning
    # if prev_centroid != 0:
    #     if centroid < (prev_centroid * 0.9):
    #         print("Warning: Person has fallen")
    # prev_centroid = centroid
    # show the output frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()