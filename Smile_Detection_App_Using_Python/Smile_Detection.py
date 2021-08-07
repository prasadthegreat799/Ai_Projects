import cv2
from random import randrange

#trained data
trained_data=cv2.CascadeClassifier('haarcascade_smile.xml')

img=cv2.imread('./images/smiling.jpg')

grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)




smile_cordinates=trained_data.detectMultiScale(grayscaled_img,scaleFactor=1.7,minNeighbors=20)
print (smile_cordinates)

for (x,y,w,h) in smile_cordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("hi prasad",img)
cv2.waitKey()


