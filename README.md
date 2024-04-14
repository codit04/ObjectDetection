# Object Detection Project with OpenCV and YOLOv3

This project showcases object detection using the YOLOv3 (You Only Look Once) an CNN implemented with OpenCV in Python. Users can perform object detection on images, videos, and live camera streams (from PC or mobile cameras using DroidCam) through a user-friendly web interface created with Streamlit.
Features

- Object detection on images, videos, and live camera streams
- Supports different input sources: local files, and live camera streams from PC or mobile
- User-friendly web interface using Streamlit for easy interaction
- Customizable object detection using pre-trained YOLOv3 weights

# Installation

1. Clone the repository:


    git clone https://github.com/codit03/ObjectDetection.git
    cd ObjectDetection

2. Install the required Python dependencies:

  
    pip install -r requirements.txt

3. Install [DroidCam](https://www.dev47apps.com/) on your mobile device:

   
DroidCam is available on the Google Play Store (for Android devices) or the App Store (for iOS devices) and on PC it's available for Windows and Linux.
   Install it and connect your mobile device to your computer via USB or Wi-Fi.


4. Download the YOLOv3 weights:


      
Download the pre-trained YOLOv3 weights from the official YOLO website by running the following command and add it to the server directory:

    wget https://pjreddie.com/media/files/yolov3.weights


# Usage
### Running the Streamlit Web Application

To start the Streamlit web application for object detection, run:
    
        streamlit run main.py

This will launch a local web server and open the application in your default web browser. From the web interface, users can perform the following actions:

- Select the input source for object detection: image, video, or live camera stream
    
- View the detected objects and bounding boxes overlaid on the input images or video frames
