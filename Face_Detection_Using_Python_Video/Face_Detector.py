import cv2 
from random import randrange 

#Loads Pre-Trained Data
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Assigning video stream as input source
img=cv2.VideoCapture(0)


while True:

        #Reading input data form img
        successfull_frame_read,frame=img.read()

        #converting image to grayscale
        grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #finding co-ordinates of faces
        face_cordinates=trained_face_data.detectMultiScale(grayscaled_img)
        print(face_cordinates)

        for (x,y,w,h) in face_cordinates:
                cv2.rectangle(frame,(x,y),(w+x,h+y),(randrange(256),randrange(256),randrange(256),2))

        #Shows image
        cv2.imshow("hi Prasad",frame)
        key=cv2.waitKey(1)

        #if Q is pressed the app will quit
        if ( key==113 or key==81):
                
                break

#Release video capture object
img.release()