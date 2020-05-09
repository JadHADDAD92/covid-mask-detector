""" Face detection using neural network
"""
from pathlib import Path

import numpy as np
from cv2 import resize
from cv2.dnn import blobFromImage, readNetFromCaffe


class FaceDetectorException(Exception):
    """ generic default exception
    """


class FaceDetector:
    """ Face Detector class
    """
    def __init__(self, prototype: Path=None, model: Path=None,
                 confidenceThreshold: float=0.6):
        self.prototype = prototype
        self.model = model
        self.confidenceThreshold = confidenceThreshold
        if self.prototype is None:
            raise FaceDetectorException("must specify prototype '.prototxt.txt' file "
                                        "path")
        if self.model is None:
            raise FaceDetectorException("must specify model '.caffemodel' file path")
        self.classifier = readNetFromCaffe(str(prototype), str(model))
    
    def detect(self, image):
        """ detect faces in image
        """
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces.append(np.array([startX, startY, endX-startX, endY-startY]))
        return faces
