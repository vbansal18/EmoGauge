import numpy as np
import cv2
from keras.models import load_model


info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+"*50, "loadin gmmodel")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)


def emotion_detect():
    found = False
    cap = cv2.VideoCapture(0)
    
    while not found:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))
    cv2.imshow('frame', frm)
    roi = roi / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))
    prediction = model.predict(roi)
    prediction = np.argmax(prediction)
    prediction = label_map[prediction]
    print("Detected Expression", prediction)
    
    cv2.destroyAllWindows()
    cap.release()

emotion_detect()
