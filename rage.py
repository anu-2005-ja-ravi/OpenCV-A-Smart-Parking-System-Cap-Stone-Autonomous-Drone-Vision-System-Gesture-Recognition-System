


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400] # Region of interest (ROI) for hand
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range for HSV (adjust if needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise reduction
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)

        # Approximate the contour to reduce number of points
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Convex hull
        hull = cv2.convexHull(approx, returnPoints=False)

        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(approx, hull)

            if defects is not None:
                count_defects = 0

                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])

                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))

                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 57 # degrees

                    # If angle < 90 degree, consider as defect
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(roi, far, 5, (0,0,255), -1)

                # Gesture interpretation
                if count_defects == 0:
                    gesture = "Fist"
                elif count_defects == 1:
                    gesture = "One Finger"
                elif count_defects == 2:
                    gesture = "Two Fingers"
                elif count_defects == 3:
                    gesture = "Three Fingers"
                elif count_defects == 4:
                    gesture = "Four Fingers"
                else:
                    gesture = "Open Hand"

                cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
        cv2.drawContours(roi, [approx], -1, (0,0,255), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
