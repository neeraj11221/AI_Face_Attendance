import cv2

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)  # or CAP_VFW

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
