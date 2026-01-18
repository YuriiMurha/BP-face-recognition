import face_recognition
import cv2

# load images
imgSample = face_recognition.load_image_file('Face-attendance-course\Basic\ImagesBasic\webcam_example.jpg')
imgSample = cv2.cvtColor(imgSample,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Face-attendance-course\Basic\ImagesBasic\seccam_example.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# find faces
faceLoc = face_recognition.face_locations(imgSample)[0]
encodeSample = face_recognition.face_encodings(imgSample)[0]
cv2.rectangle(imgSample,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # top, right, bottom, left

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
    
# compare faces
results = face_recognition.compare_faces([encodeSample], encodeTest)
faceDis = face_recognition.face_distance([encodeSample], encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)

# Save results
cv2.imwrite('Face-attendance-course\Basic\ImagesBasic\webcam_result.jpg', imgSample)
cv2.imwrite('Face-attendance-course\Basic\ImagesBasic\seccam_result.jpg', imgTest)

# show results
cv2.imshow('Sample', imgSample)
cv2.imshow('Test', imgTest)
cv2.waitKey(0)