import cv2
import numpy as np
import os
import streamlit as st

class Detector:
    def __init__(self, weights, config, labels):
        self.weights = weights
        self.config = config
        self.labels = labels
        self.net = cv2.dnn.readNet(self.weights, self.config)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [
            self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        self.colors = np.random.uniform(0, 255, size=(len(self.labels), 3))
        self.object_paths = {}  # Dictionary to store paths of detected objects
        self.is_first_frame = True  # Flag to track if it's the first frame

    def detect(self, image):
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(
            image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        sizes = []  # List to store estimated sizes of detected objects
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    # Calculate estimated size (area) of the object
                    size = w * h
                    sizes.append(size)
                    # Update object paths
                    label = str(self.labels[class_id])
                    if label not in self.object_paths:
                        self.object_paths[label] = []  # Initialize path for new object
                    self.object_paths[label].append((center_x, center_y))  # Append center coordinates

        # Draw object paths only from the second frame onwards
        if not self.is_first_frame:
            for label, path in self.object_paths.items():
                if len(path) > 1:
                    for i in range(1, len(path)):
                        cv2.line(image, path[i - 1], path[i], self.colors[self.labels.index(label)], 2)

        self.is_first_frame = False  # Update flag after processing the first frame

        # Perform non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.labels[class_ids[i]])
                size = sizes[i]  # Get estimated size of the object
                color = self.colors[class_ids[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2
                )
                detected_objects.append((label, size))  # Store label and size
        return image, detected_objects

    def detect_from_file(self, img):
        image = img
        image, detected_objects = self.detect(image)
        return image, detected_objects

    def detect_from_camera(self, camera=0):
        cap = cv2.VideoCapture(camera)
        while True:
            ret, frame = cap.read()
            frame, detected_objects = self.detect(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_from_mobile(self, ip):
        cap = cv2.VideoCapture(ip)
        while True:
            ret, frame = cap.read()
            frame, detected_objects = self.detect(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Unable to open video file.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, detected_objects = self.detect(frame)
            yield frame, detected_objects

        cap.release()