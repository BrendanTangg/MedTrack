import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

model_path = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("Stream failed :(")
        exit()

    start_time = time.monotonic()
    last_timestamp_ms = -1

    while(True):
        ret, frame = stream.read()
        if not ret:
            print("No more stream :(")
            break

        # Convert the frame from BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White color range
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([179, 45, 255])

        # Create mask for white pixels
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find white objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pill_center = None

        best_contour = None
        best_score = -1

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 20 or area > 40:
                continue

            x, y, pill_w, pill_h = cv2.boundingRect(contour)

            aspect_ratio = pill_w / pill_h

            if not (0.5 < aspect_ratio < 2.5):
                continue

            print(
                "area:", area,
                "width:", pill_w,
                "height:", pill_h,
                "aspect:", round(aspect_ratio, 2)
            )

            score = area

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is not None:
            x, y, pill_w, pill_h = cv2.boundingRect(best_contour)

            pill_x = x + pill_w // 2
            pill_y = y + pill_h // 2

            pill_center = (pill_x, pill_y)

            cv2.rectangle(frame, (x, y), (x+pill_w, y+pill_h), (0, 255, 255), 2)
            cv2.circle(frame, pill_center, 3, (0, 255, 255), -1)
            cv2.putText(frame, "Pill", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic()-start_time)*1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = latest_result.face_landmarks[0]

            mouth_indices = [61, 291, 13, 14]
            for index in mouth_indices:
                lm = face_landmarks[index]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]

            left_mouth = face_landmarks[61]
            right_mouth = face_landmarks[291]

            upper_y = int(upper_lip.y * h)
            lower_y = int(lower_lip.y * h)

            left_x = int(left_mouth.x * w)
            right_x = int(right_mouth.x * w)

            mouth_opening = abs(upper_y - lower_y)
            mouth_width = abs(right_x - left_x)

            mouth_ratio = mouth_opening / mouth_width

            if mouth_ratio > 0.1:
                mouth_status = "Mouth Open"
            else:
                mouth_status = "Mouth Closed"

            pill_in_mouth = False

            if pill_center is not None:
                pill_x, pill_y = pill_center

                if (left_x < pill_x < right_x and upper_y < pill_y < lower_y):
                    pill_in_mouth = True

            cv2.rectangle(frame, (left_x, upper_y), (right_x, lower_y), (255, 0, 255), 2)
            cv2.putText(frame, mouth_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if pill_in_mouth and mouth_ratio > 0.1:
                cv2.putText(frame, "Pill Entering Mouth", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam", frame)
        cv2.imshow("Pill Mask", mask)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
cv2.destroyAllWindows()