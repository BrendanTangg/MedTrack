import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

face_model_path = "models/face_landmarker.task"
hand_model_path = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

latest_face_result = None
latest_hand_result = None

# Create a face landmarker instance with the live stream mode:
def face_print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))
    global latest_face_result
    latest_face_result = result

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=face_print_result)

def hand_print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    global latest_hand_result
    latest_hand_result = result

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_print_result)

with FaceLandmarker.create_from_options(face_options) as face_landmarker:
  # The landmarker is initialized. Use it here.
  with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic()-start_time)*1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        face_landmarker.detect_async(mp_image, timestamp_ms)
        hand_landmarker.detect_async(mp_image, timestamp_ms)

        if latest_face_result and latest_face_result.face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = latest_face_result.face_landmarks[0]

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

            cv2.rectangle(frame, (left_x, upper_y), (right_x, lower_y), (255, 0, 255), 2)
            cv2.putText(frame, mouth_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if latest_hand_result and latest_hand_result.hand_landmarks:
            h, w = frame.shape[:2]

            for hand_landmarks in latest_hand_result.hand_landmarks:
                hand_xs = []
                hand_ys = []
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    hand_xs.append(x)
                    hand_ys.append(y)

                padding = 20

                hand_left = max(0, min(hand_xs) - padding)
                hand_right = min(w, max(hand_xs) + padding)

                hand_top = max(0, min(hand_ys) - padding)
                hand_bottom = min(h, max(hand_ys) + padding)

                hand_region = frame[hand_top:hand_bottom, hand_left:hand_right].copy()

                # Convert the frame from BGR to HSV
                hsv_frame = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
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

                best_candidate = None
                best_score = float("inf")

                for contour in contours:
                    area = cv2.contourArea(contour)
        
                    if area < 20:
                        continue
        
                    x, y, pill_w, pill_h = cv2.boundingRect(contour)

                    if pill_h == 0:
                        continue
        
                    aspect_ratio = pill_w / pill_h
        
                    if not (0.5 < aspect_ratio < 2.5):
                        continue

                    target_area = 120
                    target_aspect = 1.3

                    area_difference = abs(area - target_area)
                    aspect_difference = abs(aspect_ratio - target_aspect)

                    score = area_difference + 100 * aspect_difference

                    if score < best_score:
                        best_score = score

                        best_candidate = {
                            "x": x,
                            "y": y,
                            "w": pill_w,
                            "h": pill_h,
                            "area": area,
                            "aspect": aspect_ratio
                        }

                if best_candidate is not None:
                    x = best_candidate["x"]
                    y = best_candidate["y"]
                    pill_w = best_candidate["w"]
                    pill_h = best_candidate["h"]
        
                    full_x = hand_left + x
                    full_y = hand_top + y
        
                    pill_x = full_x + pill_w // 2
                    pill_y = full_y + pill_h // 2

                    pill_center = (pill_x, pill_y)
        
                    cv2.rectangle(frame, (full_x, full_y), (full_x + pill_w, full_y + pill_h), (0, 255, 255), 2)
                    cv2.circle(frame, pill_center, 3, (0, 255, 255), -1)
                    cv2.putText(frame, "Pill", (full_x, full_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                    cv2.rectangle(frame, (hand_left, hand_top), (hand_right, hand_bottom), (255, 0, 0))


        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
cv2.destroyAllWindows()