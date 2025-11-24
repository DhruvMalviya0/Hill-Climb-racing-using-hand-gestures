import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import math

# ==========================================
# CONFIGURATION
# ==========================================
# Camera Settings
CAM_WIDTH = 640
CAM_HEIGHT = 480
WINDOW_NAME = "Gesture Controller (Always on Top)"

# Detection Sensitivity
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

class GameController:
    def __init__(self):
        # Initialize Keyboard Controller
        self.keyboard = Controller()
        
        # State tracking to prevent key spamming
        self.left_pressed = False
        self.right_pressed = False

        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)

    def is_hand_closed(self, hand_landmarks, width, height):
        """
        Determines if a hand is closed (fist) or open.
        Uses the distance between fingertips and wrist vs PIP joints and wrist.
        """
        # Wrist is landmark 0
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y = wrist.x * width, wrist.y * height

        # Finger tips indices: Index(8), Middle(12), Ring(16), Pinky(20)
        # Finger PIP (knuckle) indices: Index(6), Middle(10), Ring(14), Pinky(18)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        folded_fingers = 0

        for tip_idx, pip_idx in zip(tips, pips):
            tip = hand_landmarks.landmark[tip_idx]
            pip = hand_landmarks.landmark[pip_idx]

            # Convert to pixel coords
            tx, ty = tip.x * width, tip.y * height
            px, py = pip.x * width, pip.y * height

            # Calculate distance to wrist
            dist_tip_wrist = math.hypot(tx - wrist_x, ty - wrist_y)
            dist_pip_wrist = math.hypot(px - wrist_x, py - wrist_y)

            # If tip is closer to wrist than the knuckle is, it's folded
            if dist_tip_wrist < dist_pip_wrist:
                folded_fingers += 1

        # Consider hand closed if at least 3 fingers are folded
        return folded_fingers >= 3

    def update_keys(self, left_active, right_active):
        """
        Updates the physical keyboard state based on hand position.
        Uses state tracking to only press/release when state changes.
        """
        # Handle Left Arrow (Brake)
        if left_active and not self.left_pressed:
            self.keyboard.press(Key.left)
            self.left_pressed = True
        elif not left_active and self.left_pressed:
            self.keyboard.release(Key.left)
            self.left_pressed = False

        # Handle Right Arrow (Gas)
        if right_active and not self.right_pressed:
            self.keyboard.press(Key.right)
            self.right_pressed = True
        elif not right_active and self.right_pressed:
            self.keyboard.release(Key.right)
            self.right_pressed = False

    def start(self):
        print("Starting Gesture Controller...")
        print("Controls: Left Side + Fist = Brake, Right Side + Fist = Gas")
        print("Open hand to release key.")
        print("Press 'q' to quit.")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 1. Pre-processing
            # Flip horizontally for mirror effect (Natural interaction)
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            # 2. Logic Calculation
            current_left_active = False
            current_right_active = False

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Get wrist position (landmark 0)
                    wrist_x = hand_lms.landmark[0].x * w
                    wrist_y = hand_lms.landmark[0].y * h

                    # Check if hand is closed (fist)
                    is_closed = self.is_hand_closed(hand_lms, w, h)

                    # Draw Skeleton
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                    # Determine Zone and Action
                    if wrist_x < w // 2:
                        # Left Zone
                        label = "OPEN"
                        color = (0, 255, 255) # Yellow
                        
                        if is_closed:
                            current_left_active = True
                            label = "BRAKE!"
                            color = (0, 0, 255) # Red
                        
                        cv2.circle(frame, (int(wrist_x), int(wrist_y)), 20, color, -1)
                        cv2.putText(frame, label, (int(wrist_x)-30, int(wrist_y)-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        # Right Zone
                        label = "OPEN"
                        color = (0, 255, 255) # Yellow
                        
                        if is_closed:
                            current_right_active = True
                            label = "GAS!"
                            color = (0, 255, 0) # Green

                        cv2.circle(frame, (int(wrist_x), int(wrist_y)), 20, color, -1)
                        cv2.putText(frame, label, (int(wrist_x)-20, int(wrist_y)-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. Execute Keyboard Commands
            self.update_keys(current_left_active, current_right_active)

            # 4. Draw HUD (Heads Up Display)
            # Center Line
            cv2.line(frame, (w // 2, 0), (w // 2, h), (50, 50, 50), 2)
            
            # Status Text overlay
            # Left Status
            color_l = (0, 0, 255) if current_left_active else (100, 100, 100)
            thickness_l = 3 if current_left_active else 1
            cv2.rectangle(frame, (10, 10), (w//2 - 10, h - 10), color_l, thickness_l)
            cv2.putText(frame, "BRAKE (FIST)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_l, 2)

            # Right Status
            color_r = (0, 255, 0) if current_right_active else (100, 100, 100)
            thickness_r = 3 if current_right_active else 1
            cv2.rectangle(frame, (w//2 + 10, 10), (w - 10, h - 10), color_r, thickness_r)
            cv2.putText(frame, "GAS (FIST)", (w//2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_r, 2)

            # 5. Display
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GameController()
    app.start()
