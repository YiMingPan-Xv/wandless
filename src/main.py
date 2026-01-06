import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

import shortcuts as sc
import config as cf

model_path = str(Path(__file__).parents[1] / "tasks" / "hand_landmarker.task")

current_result = None

# A list of tuples defining which landmarks are connected
# Used to draw manually the skeleton of the hand
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)     # Pinky
]


class GestureManager:
    def __init__(self):
        self.history = deque(maxlen=cf.HISTORY_LENGTH)
        self.last_trigger_time = 0
        self.last_frame_time = 0

    def process_landmarks(self, hand_landmarks):
        idx_tip = hand_landmarks[8]

        other_fingers = [hand_landmarks[4], hand_landmarks[12], hand_landmarks[16], hand_landmarks[20]]

        lowest_dist = min([
            np.sqrt((idx_tip.x - other_tip.x)**2 + (idx_tip.y - other_tip.y)**2)
            for other_tip
            in other_fingers
            ])

        if time.time() - self.last_frame_time > cf.MAX_TIME_GAP:
            self.history.clear()

        if lowest_dist > cf.FINGER_PROXIMITY_THRESHOLD:
            x = idx_tip.x
            y = idx_tip.y
            if len(self.history) == 0:
                if (
                    x < cf.START_MARGIN
                        or x > (1 - cf.START_MARGIN)
                        or y < cf.START_MARGIN
                        or y > (1 - cf.START_MARGIN)):
                    return None  # The hand was detected in the margin, outside the detection box

            self.history.append((x, y))
            self.last_frame_time = time.time()
        else:
            return None

        if len(self.history) == cf.HISTORY_LENGTH and (time.time() - self.last_trigger_time) > cf.COOLDOWN_TIME:
            dx = self.history[-1][0] - self.history[0][0]
            dy = self.history[-1][1] - self.history[0][1]

            if abs(dx) > cf.X_SWIPE_THRESHOLD and abs(dx) > abs(dy):
                self.last_trigger_time = time.time()
                self.history.clear()
                return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
            elif abs(dy) > cf.Y_SWIPE_THRESHOLD and abs(dy) > abs(dx):
                self.last_trigger_time = time.time()
                self.history.clear()
                return "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"
            else:
                self.history.popleft()
        return None


manager = GestureManager()


def process_results(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # Note: vision.HandLandmarkerResult is detected as variable, but the variable itself points to the class
    # Linters might raise a warning
    global current_result
    current_result = result

    if result.hand_landmarks:
        gesture = manager.process_landmarks(result.hand_landmarks[0])
        if gesture:
            print(f"GESTURE DETECTED: {gesture}")
            if gesture == "SWIPE_LEFT":
                sc.left_gesture()
            elif gesture == "SWIPE_RIGHT":
                sc.right_gesture()
            elif gesture == "SWIPE_UP":
                sc.up_gesture()
            elif gesture == "SWIPE_DOWN":
                sc.down_gesture()


def draw_hand_on_frame(rgb_image, hand_landmarker_result):
    """
    Draws the hand skeleton directly using OpenCV.

    It also draws the box in which the detection is triggered.
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    point1 = (int(width * cf.START_MARGIN), int(height * cf.START_MARGIN))
    point2 = (int(width * (1 - cf.START_MARGIN)), int(height * (1 - cf.START_MARGIN)))

    cv2.rectangle(
        annotated_image,
        point1,
        point2,
        (0, 0, 255),
        2
        )

    minimum_distance_proximity = np.sqrt(width**2 + height**2) * cf.FINGER_PROXIMITY_THRESHOLD

    # Iterate over all detected hands
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:

            pixel_landmarks = []
            for landmark in hand_landmarks:
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                pixel_landmarks.append((px, py))

            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                cv2.line(annotated_image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (224, 224, 224), 2)

            for i, (px, py) in enumerate(pixel_landmarks):
                if i == 8:
                    other_fingers = [pixel_landmarks[4], pixel_landmarks[12], pixel_landmarks[16], pixel_landmarks[20]]
                    lowest_dist = min([
                        np.sqrt((px - ox)**2 + (py - oy)**2)
                        for (ox, oy)
                        in other_fingers
                        ])
                    if lowest_dist > minimum_distance_proximity:
                        cv2.circle(annotated_image, (px, py), 4, (0, 255, 0), -1)
                        continue
                cv2.circle(annotated_image, (px, py), 4, (0, 165, 255), -1)

    return annotated_image


options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=process_results,
    num_hands=1)

try:
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cam = cv2.VideoCapture(0)

        while cam.isOpened():
            ret, frame = cam.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)

            landmarker.detect_async(mp_image, timestamp_ms)

            # Draw skeleton if results exist
            if current_result:
                frame = draw_hand_on_frame(frame, current_result)

            cv2.putText(frame, "Wandless", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Hand Landmarker', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
except KeyboardInterrupt:
    print("Ctrl+C detected.\nNote: You can also close the program by focusing on the webcam window and pressing ESC.")
finally:
    cam.release()
    cv2.destroyAllWindows()
