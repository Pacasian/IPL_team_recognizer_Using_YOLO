
import numpy as np
import argparse
import imutils
import time
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

def count_nonblack_np(img):
    return img.any(axis=-1).sum()


def boun(imge):
    imag=cv2.cvtColor(imge, cv2.COLOR_BGR2HSV)
   #red----[17, 15, 100], [50, 56, 200]
    boundaries = [
        ([110,50,50],[130,255,255]), #blue
        ([25, 146, 190], [96, 174, 250]) #yellow
        ]
    i = 0
    #Blue code 
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(imag, lower, upper)
        output = cv2.bitwise_and(imag, imag, mask = mask)
        tot_pix = count_nonblack_np(imag)
        color_pix = count_nonblack_np(output)
        ratio = color_pix/tot_pix
        if ratio > 0.2 and i == 0:
            return 'mumbai'
        #yellow
        elif ratio > 0.01 and i == 1:
            return 'chennai'
        i += 1
    return "not known person"


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


while True:
	
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

	# if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
			
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

			
            if confidence > args["confidence"]:
				
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

				
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
            frames=text
            if (text.split(":")[0])=="person":
                crop_img = frame[y:y+h, x:x+w]
                frames=boun(crop_img)
            cv2.putText(frame, frames, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
vs.release()
