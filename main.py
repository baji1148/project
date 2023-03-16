import cv2

face_cascade = cv2.CascadeClassifier("C:/Users/SHAIK BAJI BABA/3D Objects/Mini_project/haarcascade/haarcascade_frontalface_default.xml")
#left_eye_cascade = cv2.CascadeClassifier("C:/Users/SHAIK BAJI BABA/3D Objects/Mini_project/haarcascade/lefteyetest.xml")
#right_eye_cascade = cv2.CascadeClassifier("C:/Users/SHAIK BAJI BABA/3D Objects/Mini_project/haarcascade/righteyetest.xml")


eye_cascade = cv2.CascadeClassifier("C:/Users/SHAIK BAJI BABA/3D Objects/Mini_project/haarcascade/fulleye.xml")
i=0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    #left_eye = left_eye_cascade.detectMultiScale(frame)
    #right_eye = right_eye_cascade.detectMultiScale(frame)

    #print("faces :",faces)
    #print("left_eye :",left_eye)
    #print("right_eye :",right_eye)
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) which is the face area
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_color, 1.1, 3)

        # Iterate over each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Check if the eye is on the left or right side of the face
            if ex + ew/2 < w/2:
                # This eye is on the left side of the face
                left_eye = roi_color[ey:ey+eh, ex:ex+ew]
                left_eye = cv2.resize(left_eye,(500,500))
                cv2.imshow('Left Eye', left_eye)
                cv2.imwrite("./pics/left_eye/img"+str(i)+".png",left_eye)
            else:
                # This eye is on the right side of the face
                right_eye = roi_color[ey:ey+eh, ex:ex+ew]
                right_eye = cv2.resize(right_eye,(500,500))
                cv2.imshow('Right Eye', right_eye)
                cv2.imwrite("./pics/right_eye/img"+str(i)+".png",right_eye)
    i+=1

   

    
    cv2.imshow("frame",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
