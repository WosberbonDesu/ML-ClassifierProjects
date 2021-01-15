import cv2
import matplotlib.pyplot as plt

einstein = cv2.imread("here picture name",0)
plt.figure(),plt.imshow(einstein, cmap = "gray"),plt.axis("off"),plt.show()


# classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y), (x+w, y+h),(255,255,0),10)

plt.figure(),plt.imshow(einstein, cmap = "gray"),plt.axis("off"),plt.show()




barce = cv2.imread("PİCTURE HERE",0)
plt.figure(),plt.imshow(barce, cmap = "gray"),plt.axis("off"),plt.show()



face_rect = face_cascade.detectMultiScale(barce, minNeighbors = 10)


for (x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y), (x+w, y+h),(255,255,0),10)

plt.figure(),plt.imshow(barce, cmap = "gray"),plt.axis("off"),plt.show()


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 10)
    for (x,y,w,h) in face_rect:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,0),10)
    cv2.imshow("face detect",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()

cv2.destroyAllWindows()