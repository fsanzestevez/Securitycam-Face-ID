# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:55:19 2020

@author: Fran
"""

import face_recognition as fr
import cv2
import numpy as np
import pickle

def face_recog_train():
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_'\
                                         'frontalface_alt2.xml')
    
    
    fran_image = fr.load_image_file('data/images/face_recognition/fran.JPG')
    ronaldo_image = fr.load_image_file('data/images/face_recognition/ronaldo.JPG')
    obama_image = fr.load_image_file('data/images/face_recognition/obama.jpg')
    
    
    fran_encoding = fr.face_encodings(fran_image)[0]
    ronaldo_encoding = fr.face_encodings(ronaldo_image)[0]
    obama_encoding = fr.face_encodings(obama_image)[0]

    
    
    known_face_encodings = [
        fran_encoding,
        ronaldo_encoding,
        obama_encoding
    ]
    known_face_names = [
        "Francisco",
        "Ronaldo",
        "Obama"
    ]
    
    
    with open('data/training/encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f) 
    
    with open('data/training/names.pkl', 'wb') as f:
        pickle.dump(known_face_names, f) 


