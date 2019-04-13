# **IPL_team_recognizer_Using_YOLO**

**Snippet**
Basically this project is to identify the teams in IPL(Indian Premier League) through the concept of computer-vision.
So, we used the YOLO object detection algorithm through COCO dataset for identifying the objects in the frame and
then masked the area where it detects person and using colorstats we differentiate each team.
Currently , CSK(Chennai Super Kings) and MI(Mumbai Indians) are being detected with average precision.(further teams 
can be added using respective HSV colour codes)

**Important**
Visit the site and download the dataset and save it in the yolo-coco folder 
**https://github.com/nightrome/cocostuff**
this is an essential part because the coco dataset will be having the labels and weights of each and every object(most of all 
real world objects) thus the yolo algorithm compare the extracted features(weights) with the predefined ones.

**Requirements**
**Hardware:**
- -> Nvidia or AMD grapics (** it can be also achieved through CPU potentials**)
**Software:**
- ->Any python based environment (preferibly Anaconda)
- ->installed opencv(preferbily Opencv3 rather than opencv4)
- ->installed imutils(for installing ** pip install --upgrade imuilts**)

**Program with Image Input:**
This project provides sample image-based team detection so, if you want to check the working of this module , compile the 
**yolo.py** ,for testing purpose you can import images to the **image folder** of your favourate match and call that particular image through 
image through its URL.
The command for this module is :
**python yolo.py --image image/your_image_name --yolo yolo-colo**

**Program with Video Input:**
This project also provides sample video-based team detection so, if you want to check the working of this module , compile the 
**yolo_video.py** ,for testing purpose you can import the video to the **input folder** of your favourate match and call that particular 
video through its URL.
The command for this module is :
**python yolo_video.py --input input/your_video_name --output output/any_name.avi --yolo yolo-colo**
the output file must be of avi extension

**Credits**
https://pjreddie.com/darknet/yolo/
https://www.pyimagesearch.com/
https://github.com/priya-dwivedi/
