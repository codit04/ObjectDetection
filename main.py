from Server.detector import Detector
import streamlit as st
import cv2
import numpy as np
import tempfile

#detector setup
weights = 'Server/yolov3.weights'
config = 'Server/yolov3.cfg'
labels = open('Server/coco.names').read().strip().split('\n')
detector = Detector(weights, config, labels)


st.title('Object Detection')
st.write('This is an object detection app using YOLOv3')

#add a dropdown menu to select an option to upload an image or video or use the camera
option = st.selectbox('Choose an option', ['Image', 'Video', 'PC Camera', 'Mobile Camera'])

if option == 'Image':
    uploaded_file = st.file_uploader('Choose an image',type=['jpg','png'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        detector.detect_from_file(cv2.imread(tfile.name))

elif option == 'Video':
    uploaded_file = st.file_uploader('Choose a video',type=['mp4'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        detector.detect_from_video(cv2.VideoCapture(tfile.name))

elif option == 'PC Camera':
    detector.detect_from_camera()

elif option == 'Mobile Camera':
    detector.detect_from_mobile('http://192.168.1.8:4747/video')


else:
    st.write('Invalid option')
