import cv2
import numpy as np
import streamlit as st

class YOLO:
    def __init__(self, config_path, weights_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = open(classes_path).read().strip().split("\n")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers().flatten()]

        # Debug statements
        print("Layer Names:", self.layer_names)
        print("Unconnected Out Layers:", self.output_layers)

        #st.write("Layer Names:", self.layer_names)
        #st.write("Unconnected Out Layers:", self.output_layers)

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return outs
