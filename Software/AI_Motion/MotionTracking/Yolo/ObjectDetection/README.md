# 2D Object Tracker #

Daniel Bisig - Instituto Stocos, Spain - daniel@stocos.com, Zurich University of the Arts, Switzerland - daniel.bisig@zhdk.ch

### Overview ###

A simple wrapper for the Yolo V8 object tracker. This wrapper employs Yolo for tracking the bounding boxes together with class labels for different object types in a single live video image.

## OSC-Communication

The tracker communicates with other software applications by using the OSC (Open Sound Control) protocol. During tracking, the tracker sends for each detected object the class index of the object, the confidence score of the class index, and the minimum and maximum 2D positions of the bounding box corners.  

#### The following OSC messages are sent by the tracker:

- /object/boxX/bbox <Int> <Float> <Float> <Float> <Float> <Float>: sends information about each tracked object. The "X" in the message translates into the index of the tracked object. The message parameters are as follows: first parameter (int) for the class label, second parameter (float) for the class confidence, third to sixth parameter for the x y x y  coordinates the the bounding box corners. 

The code has been tested on Windows and MacOS. Anaconda environments for these two operating systems are provided as part of this repository. 