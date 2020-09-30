import cv2

class face_eye(object):
    def __init__(self,face_cascade_path="haarcascade_frontalface_alt.xml",eye_cascade_path = "haarcascade_eye.xml"):
        self.face_cascade =  cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def main(self,cap):
        self.camera(cap)

    #モザイクをかけている。
    def mozaic(self,faces,img,ratio=0.05):
        for x,y,w,h in faces:
            small = cv2.resize(img[y: y+h, x: x+w], None,fy=ratio, fx=ratio,interpolation=cv2.INTER_NEAREST)
            img[y: y + h, x: x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return img

    #平滑フィルタで顔を綺麗にしている。
    def kirei(self, faces, img):
        for x, y, w, h in faces:
            face = cv2.resize(img[y: y+h, x: x+w], None,fy=1, fx=1,interpolation=cv2.INTER_NEAREST)
            face = cv2.bilateralFilter(face, 9, 75, 75)
            img[y: y + h, x: x + w] = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        return img

    def camera(self,cap):
        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            self.mozaic(faces,img,ratio=0.05)
            #self.kirei(faces, img)
            cv2.imshow('video image', img)
            key = cv2.waitKey(10)
            if key == 27:  # ESCキーで終了
                break

if __name__ == "__main__":
    face_eye.main(face_cascade_path="haarcascade_frontalface_alt.xml",eye_cascade_path = "haarcascade_eye.xml")
