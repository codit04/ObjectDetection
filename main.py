import streamlit as st
import cv2
import numpy as np
import tempfile
from Server.detector import Detector

# Detector setup
weights = 'Server/yolov3.weights'
config = 'Server/yolov3.cfg'
labels = open('Server/coco.names').read().strip().split('\n')
detector = Detector(weights, config, labels)

st.title('Object Detection')
st.write('This is an object detection app using YOLOv3')

# Add a dropdown menu to select an option to upload an image or video or use the camera
option = st.selectbox('Choose an option', ['Image', 'Video', 'PC Camera', 'Mobile Camera'])

if option == 'Image':
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'png'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image = cv2.imread(tfile.name)
        detected_image, detected_objects = detector.detect_from_file(image)
        st.image(detected_image, channels="BGR")
        st.write("Detected Objects:")
        for obj, size in detected_objects:
            st.write(f"- {obj}: Estimated Size {size} pixels squared")

elif option == 'Video':
    uploaded_file = st.file_uploader('Choose a video', type=['mp4'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        video_generator = detector.detect_from_video(video_path)
        first_frame = True
        for frame, detected_objects in video_generator:
            st.image(frame, channels="BGR")
            st.write("Detected Objects:")
            for obj, size in detected_objects:
                st.write(f"- {obj}: Estimated Size {size} pixels squared")
            if first_frame:
                first_frame = False
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

elif option == 'PC Camera':
    detector.detect_from_camera()

elif option == 'Mobile Camera':
    detector.detect_from_mobile('http://192.168.221.180:4747/video')

else:
    st.write('Invalid option')