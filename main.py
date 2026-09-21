import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import time

face_model_path = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_face_result = None

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


model = YOLO("medtrack.pt")

with FaceLandmarker.create_from_options(face_options) as face_landmarker:
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

        results = model.predict(
            source=frame,
            conf=0.8,
            device=0,
            verbose=False
        )

        display_frame = results[0].plot()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic()-start_time)*1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        face_landmarker.detect_async(mp_image, timestamp_ms)

        if latest_face_result and latest_face_result.face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = latest_face_result.face_landmarks[0]

            mouth_indices = [61, 291, 13, 14]
            for index in mouth_indices:
                lm = face_landmarks[index]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)

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

            cv2.rectangle(display_frame, (left_x, upper_y), (right_x, lower_y), (255, 0, 255), 2)
            cv2.putText(display_frame, mouth_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Webcam", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
cv2.destroyAllWindows()