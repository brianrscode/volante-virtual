import cv2
import numpy as np
from math import degrees, acos
from PoseDetector import PoseDetector


cap = cv2.VideoCapture(2)
detector = PoseDetector()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.find_pose(frame)  # Detector de pose
    position_landmarks = detector.find_position_landmarks(frame, draw=False)  # Detector de landmarks

    if len(position_landmarks) != 0:
        # Puntos del triángulo
        p1 = np.array(position_landmarks[21][1:])  # pulgar izquierdo
        p2 = np.array(position_landmarks[22][1:])  # pulgar derecho
        p3 = np.array([p2[0], p1[1]])

        # Partes del triángulo
        hipotenusa, frame = detector.distance_points(p1, p2, frame, draw_d=True, color=(255, 255, 0))
        cateto, frame = detector.distance_points(p1, p3, frame, draw_d=True)

        angulo = degrees(acos(cateto / hipotenusa))
        if p1[1] < p2[1]:  # Si el punto 1 es inferior en "y" al punto 2
            angulo = -angulo
        cv2.putText(frame, str(int(angulo)), (p1[0] + 10, p1[1] + 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Detección de ángulo positivo o negativo

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()