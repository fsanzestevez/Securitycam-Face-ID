# Security Camera  
  
In this repository we develop a security camera with face recognition using the libraries Open-CV and Face-recognition.

## Structure of the repository

The project, even though it is a small one, it is organized like if it were a full project. It has different folders and sub-folders which contain different information. The structure is as follows:

**security-camera** main folder.  
|-- **data** folder where data is contained  
|---- **cascades** folder with the cascades from open-cv (you can access them  directly from open-cv, but I like having them in the project's folder.  
|---- **images** folder where we store the images to train the face-recognition models.  
|------ **face_recognition** images for the library face-recognition (it only needs one of each person).  
|------ **face_train** images for the open-cv face classifier, it needs many labeled images to train the model.  
|---- **training** folder to store the already trained models.  
|-- **src** folder with all codes of the project.  
|--**main.py** main file of the project.

## Logic of the Project

The idea of the project is to develop a security camera that detects movement in the image and, if possible recognize the intruder. So, in order to accomplish this, what we do is save the first frame of the video for reference, and then every single frame that follows compare it to the first one. If the two frames have noticeable differences, then we mark it as an intruder.
To compare the frames we first need to transform the frames:  
First, we convert the frame to gray scale, then we apply a Gaussian transform with a kernel, we obtain the absolute difference between frames and apply a threshold. 
The threshold is used to push to the limits the differences, so where there is a noticeable difference we map it to completely white, if the difference isn't noticeable, we map it to black. 
We find the contours in the picture where there is large white chunks so we can add a rectable in the original frame with open-cv to mark it as the intruder or movement. 

Parallel to this, we also run the face recognition algortihm. In my experience, I have tried both, the open-cv face classifier, and the face-recognition classifier, and what worked best for me was a mixture of both: open-cv detects the areas where the faces are, but face-recognition is more accurate to detect who it is, so that is what I did:
I first analyze the frame with open-cv and the frontal-face cascade to locate the coordinates where the face is, and pass this coordinates to the face-recognition model to identify the intruder.

Since we have now signals of when there is movement or when we recognize someone, it is easy to continue the project and store only the part of the video where there is an intruder and send this piece of video via email, or message.
 
