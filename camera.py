import hitomi
import cv2

cap = cv2.VideoCapture(0)
camera = hitomi.face_eye(face_cascade_path="haarcascade_frontalface_alt.xml",eye_cascade_path = "haarcascade_eye.xml").main(cap)
cap.release()
cv2.destroyAllWindows()#
