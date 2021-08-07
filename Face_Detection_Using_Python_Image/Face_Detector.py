import cv2
from random import randrange


#Loads Pre-Trained Data
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to Detect Face Data
img=cv2.imread('./images/rdj2.jpg')



#converting image to grayscale
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#detect face
face_cordinates=trained_face_data.detectMultiScale(grayscaled_img)
print(face_cordinates)

#Draw rectangle
for (x,y,w,h) in face_cordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),5)

#shows image
cv2.imshow("Ai Face Detector Project",img)

#waits until we press any key
cv2.waitKey()

print ("code completed")