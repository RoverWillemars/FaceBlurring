import cv2
import os

directory = "FaceBlurring/DropImagesHere/"
listOfImageDir = []

#Load Pre-trained face cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

for file in os.listdir(directory):
        listOfImageDir.append(file)

for items in listOfImageDir:
    image = cv2.imread('FaceBlurring/DropImagesHere/' + str(items), -1)
    grey_scale_image = cv2.imread('FaceBlurring/DropImagesHere/' + str(items), 0)
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)

    #Detect the faces, Change the 1.1 & 3 to make the face detection more/less sensitive. Check documentation for arguments.
    faces = face_cascade.detectMultiScale(grey_scale_image, 1.1, 3)
    for (x, y, w, h) in faces:
        ROI = image[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(ROI, [51, 51], 0)
        image[y:y+h, x:x+w] = blur

        #Include line under if you want to have a rectangle drawn around detected faces
        #rect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.imwrite('FaceBlurring/BlurredOutput/blurred_' + str(items), image)