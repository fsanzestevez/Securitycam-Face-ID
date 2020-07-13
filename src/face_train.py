# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:01:45 2020

@author: Fran
"""


import os
from PIL import Image
import numpy as np
import cv2
import pickle

def face_train():
    face_cascade = cv2.CascadeClassifier('data/cascades/'\
                                         'haarcascade_frontalface_alt2.xml')
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    image_dir = 'data/images/face_train'
    
    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('png') or file.endswith('jpg'):
                path = os.path.join(root, file)
                label = os.path.basename(root)\
                    .replace(' ', '-').lower()
                print(path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]
                # print(label, file)
                pil_image = Image.open(path).convert('L') # grayscale
                image_array = np.array(pil_image, 'uint8') # np array
                # print(image_array)
                faces = face_cascade.detectMultiScale(image_array)
                
                for (x,y,w,h) in faces: 
                    x_train.append(image_array[y:y+h, x:x+w])
                    y_labels.append(id_)
    
    with open('data/training/labels.pkl', 'wb') as f:
        pickle.dump(label_ids, f) 
                   
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('data/training/trainer.yml')