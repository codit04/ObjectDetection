import cv2
import numpy as np
import os


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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.labels[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2
                )
                detected_objects.append(label)
        return image, detected_objects

    def detect_from_file(self, img):
        image = img
        image, detected_objects = self.detect(image)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return detected_objects

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
        return detected_objects

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
        return detected_objects

    def detect_from_video(self, video):
        cap = video
        while True:
            ret, frame = cap.read()
            frame, detected_objects = self.detect(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return detected_objects
