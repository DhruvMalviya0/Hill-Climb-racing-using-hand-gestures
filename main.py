import cv2
import time
import mediapipe as mp
import pyautogui

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp_tasks.vision.RunningMode

# ---------------------- MODEL SETUP ----------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# ---------------------- CAMERA SETUP ----------------------
cap = cv2.VideoCapture(1)   # try 0, 1, or 2 depending on laptop
cap.set(3, 640)
cap.set(4, 480)

prev_time = 0

# ---------------------- MAIN LOOP ----------------------
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Use VIDEO mode → detect with timestamp
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    # Process detections
    if result.hand_landmarks:
        for idx, hand in enumerate(result.hand_landmarks):

            # Draw landmarks
            for lm in hand:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # Finger tip logic
            tip_ids = [4, 8, 12, 16, 20]
            fingers = []

            # Thumb
            if hand[4].x < hand[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for tip in [8, 12, 16, 20]:
                if hand[tip].y < hand[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)

            # ----------------- CONTROL LOGIC -----------------
            if idx == 0:  # Left hand → BRAKE
                if total == 5:
                    pyautogui.keyDown("left")
                    cv2.putText(frame, "LEFT HAND: BRAKE", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    pyautogui.keyUp("left")

            elif idx == 1:  # Right hand → ACCELERATE
                if total == 0:
                    pyautogui.keyDown("right")
                    cv2.putText(frame, "RIGHT HAND: ACCELERATE", (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    pyautogui.keyUp("right")

    # FPS counter
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 0.00001))
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Gesture Control (Both Hands)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
