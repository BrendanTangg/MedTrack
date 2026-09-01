import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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

    start_time = time.time()

    while(True):
        ret, frame = stream.read()
        if not ret:
            print("No more stream :(")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time()-start_time)*1000)
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

            cv2.putText(frame, mouth_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
cv2.destroyAllWindows()