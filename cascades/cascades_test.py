import cv2

img = cv2.imread('00.png',1)#读取一张图片

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#将图片转化成灰度

face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
face_cascade.load('C:/Users/wmslab\Desktop/20200901_TelloDrone/0Tello-Face-Recognition-master/face_recognition/cascades/haarcascade_frontalcatface.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow('img',img)
cv2.waitKey()