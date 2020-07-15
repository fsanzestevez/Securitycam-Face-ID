

import cv2
import time
import datetime
import pickle
import random
import face_recognition
import numpy as np


class SecurityCam():
    def __init__(self, webcam='default'):
        self.webcam = 0
        if webcam == 'external':
            self.webcam = 1
        self.face_cascade = cv2.CascadeClassifier('data/cascades/' \
                                         'haarcascade_frontalface_alt2.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('data/training/trainer.yml')
        # label_ids = {'persons_name': 1}
        with open('data/training/labels.pkl', 'rb') as f:
            labels = pickle.load(f)   
        # Reverse to Id -> name
        self.label_ids = {v:k for k,v in labels.items()}
        with open('data/training/encodings.pkl', 'rb') as f:
            self.encodings = pickle.load(f) 
        with open('data/training/names.pkl', 'rb') as f:
            self.names = pickle.load(f) 
    
    def faceRecognition(self, frame):
        
        faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.5, 
                                              minNeighbors=5)
        
        faces_list = []
        for (x, y, w, h) in faces:
            id_, conf = self.recognizer.predict(frame[y:y+h, x:x+w])

            faces_list.append([x, y, w, h, id_])

        if len(faces_list) > 0:
            return [True, faces_list]
        else:
            return [False, faces_list]
            
            
    def advanced_face_rec(self, frame):
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        faces = self.face_cascade.detectMultiScale(frame,
                                                    scaleFactor=1.5, 
                                                    minNeighbors=5)

        faces_list = []
        face_locations2 = []
        found = False
        for (x, y, w, h) in faces:
            found = True
            # l = x; t=y; r=x+w, b = y+h
            face_locations2 = [(int(round(y/4)), int(round((x+w)/4)),
                                    int(round((y+h)/4)), int(round(x/4)))]

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = small_frame[:, :, ::-1]
            # face_locations = face_recognition.face_locations(rgb_frame)

            face_encoding = face_recognition.face_encodings(rgb_frame,
                                                            face_locations2)[0]

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.encodings,
                                                     face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.encodings,
                                                            face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.names[best_match_index]

            face_names.append(name)
            # Display the results
            # t:top, r:right, b:bottom, l: left
            # l = x; t=y; r=x+w, b = y+h
            for (t, r, b, l), name in zip(face_locations2, face_names):
                # Scale back up face locations since the frame we detected in
                # was scaled to 1/4 size
                t *= 4
                r *= 4
                b *= 4
                l *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (l, b - 35), (r, b), (0, 0, 255),
                              cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (l + 6, b - 6), font,
                            0.75, (255, 255, 255), 1)

        return frame, found
            
    def detectMotion(self):
        video = cv2.VideoCapture(self.webcam) # Default webcam
        time.sleep(1)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('data/output.avi', fourcc=fourcc, fps=20,
                                frameSize=(640,480))
        video_mask = cv2.VideoWriter('data/output_mask.avi', fourcc=fourcc, fps=20,
                                    frameSize=(640,480))
        video_delta = cv2.VideoWriter('data/output_delta.avi', fourcc=fourcc, fps=20,
                                      frameSize=(640,480))
        first_frame = None # Start-off the program with an empty frame
        process_this_frame = True
        while True:
            frame = video.read()[1] # first value of the function is a boolean
            text = '' 
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            # Convert frame to gray scale. Easier to make the threshold with
            
            gaussian = cv2.GaussianBlur(gray, (23, 23), 0)
            # Gaussian conversion with a kernel (filter) of 23x23. 
            # The 0 represents the standard deviation of the width and height.
            # frame = imutils.resize(frame, width=500)
            frame_delta = cv2.absdiff(first_frame, gaussian)
            # Abs difference between frames.
            thresh = cv2.threshold(frame_delta, 25, 255,
                                    cv2.THRESH_BINARY)[1]
            
            dilated_frame = cv2.dilate(thresh, None, iterations=2)
            
            # blurred = cv2.blur(gaussian, (5, 5))
            # Simplify the frame with a kernel of 5x5. 
            
            if first_frame is None:
                first_frame = gaussian
                # We use the first frame to detect when there is any movement 
                # in the recording. The motion will be calculated as a 
                # difference between the first frame and the current frame.
            else:
                pass
            
            if process_this_frame:
                        frame, faces = self.advanced_face_rec(frame)
            
            if faces is False:
                contours = cv2.findContours(dilated_frame.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)[0]
                
                for c in contours:
                    if cv2.contourArea(c) > 800:
                        (x, y, w, h) = cv2.boundingRect(c)
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)
                        # Set rectangle on image 'frame', in the position (x,y)  
                        # and expands up to (x+w, y+h). Uses the color 
                        # (0, 255, 0) with a thickness of 2
                        
                        text = 'Intruder Alert: Not recognizable' 
                        # Add the text in which we tell
                        # the viewer there is an intruder in the frame
                        check = False
                        # (check, coord_list) = self.faceRecognition(gray)
                        if check:
                            for face in coord_list:
                                (x, y, w, h, id_) = face
                                cv2.rectangle(frame, (x, y), (x + w, y + h),
                                              (255, 0, 0), 2)
                                text += ': ' + self.label_ids[id_]
            
            else:
                text = 'Intruder Alert'
            # process_this_frame = not process_this_frame    
            # Adding text to the frames.
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(frame, text, (10, 20), font, 0.5, (0, 0, 255), 2)
            
            cv2.putText(frame, datetime.datetime.now() \
                        .strftime('%A %d %B %Y %I:%M:%S%p'),
                        (10, frame.shape[0] - 10), font, 0.35, (0, 0, 255), 1)

            video_out.write(frame)
            video_mask.write(dilated_frame)
            video_delta.write(frame_delta)
            cv2.imshow('Threshold (foreground mask)', dilated_frame)
            cv2.imshow('Frame_delta', frame_delta)
            cv2.imshow('Security Feed', frame)
            
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        
