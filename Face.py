# Detects and recognizes faces using 
import cv2
import numpy as np
import face_recognition
import os

print('Starting')
path = 'Faces'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace,faceloc in zip(encodesCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
            cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,255,255),2)

    cv2.imshow('Cam', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
