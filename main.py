# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:45:04 2020

@author: Fran
"""

import time
from src.securityCam import SecurityCam
import src.face_train as ft
import src.face_recog as fr

if __name__=='__main__': 
    # time.sleep(1)
    cam = SecurityCam()
    cam.detectMotion()
    # ft.face_train()