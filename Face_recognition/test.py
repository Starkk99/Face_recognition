# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:41:58 2020

@author: Pranav
"""

import face_recognition
import cv2
import numpy as np

# it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
pranav_image = face_recognition.load_image_file("Pranav4.jpg")
face_landmarks = face_recognition.face_encodings(pranav_image)
print(face_landmarks)