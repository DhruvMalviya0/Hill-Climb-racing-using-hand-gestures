"""
hand_control_hcr.py

Hill Climb Racing - Camera control (Option B: Palm open / Fist detection)
Requirements:
    pip install mediapipe opencv-python pyautogui

Usage:
    python hand_control_hcr.py
    Optional args:
        --cam N      (camera index, default 0)
        --min_det    (min_detection_confidence, default 0.6)
        --min_track  (min_tracking_confidence, default 0.6)
        --frames     (frames threshold for smoothing, default 6)

How it works (short):
    - Detect hands with MediaPipe Hands (max 2 hands)
    - For each hand:
        * determine handedness (Left/Right)
        * compute finger states (open/folded) using landmark comparisons
        * if palm open (>=4 fingers open) -> press and hold corresponding arrow key
        * if fist (<=1 fingers open) -> release corresponding arrow key
    - Debounce: gestures must persist for N frames before acting
    - q or ESC quits
"""

import time
import argparse
from collections import deque, defaultdict

import cv2
import mediapipe as mp
import pyautogui

# ---------- Configurable constants ----------
DEFAULT_CAM_INDEX = 0
DEFAULT_MIN_DET = 0.6
DEFAULT_MIN_TRACK = 0.6
DEFAULT_FRAME_THRESHOLD = 6  # number of consecutive frames required to commit gesture change
# --------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices used by Mediapipe Hands
TIP_IDS = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky
PIP_IDS = [2, 6, 10, 14, 18]     # approximate pip/knuckle for thumb use 2 (IP/MCP side)


def fingers_state(hand_landmarks, handedness_str):
    """
    Determine which fingers are extended.

    Returns:
        list of bool [thumb, index, middle, ring, pinky]
    Heuristics:
        - For index..pinky: compare y of tip vs pip (screen coordinates: smaller y is up)
          If tip_y < pip_y => finger is extended (open).
        - For thumb: compare x coordinates because thumb extends sideways.
          For right hand: thumb is extended if tip_x > ip_x (to the right)
          For left  hand: tip_x < ip_x (to the left)
    """
    lm = hand_landmarks.landmark
    fingers = [False] * 5

    # Index, middle, ring, pinky: tip vs pip by y
    for i, (tip_id, pip_id) in enumerate(zip(TIP_IDS[1:], PIP_IDS[1:]), start=1):
        fingers[i] = lm[tip_id].y < lm[pip_id].y  # True if finger is up (smaller y is higher on image)

    # Thumb logic (use x coordinate)
    # thumb tip is TIP_IDS[0] (4), use pip-like point PIP_IDS[0] (2) for comparison
    thumb_tip_x = lm[TIP_IDS[0]].x
    thumb_ip_x = lm[PIP_IDS[0]].x
    if handedness_str.lower().startswith("right"):
        fingers[0] = thumb_tip_x > thumb_ip_x + 0.02  # small margin avoids noise
    else:
        fingers[0] = thumb_tip_x < thumb_ip_x - 0.02

    return fingers  # [thumb, index, middle, ring, pinky]


class KeyController:
    """
    Keeps track of keyDown/keyUp state and debouncing logic.
    Designed to be simple and robust.
    """
    def __init__(self, frame_threshold=DEFAULT_FRAME_THRESHOLD):
        # track desired state candidates over last N frames
        self.frame_threshold = frame_threshold
        # counters: how many consecutive frames the desired state persisted
        self.counters = defaultdict(lambda: 0)
        # committed state (what keys are currently held)
        self.committed = {
            'left': False,   # brake -> left arrow
            'right': False   # accelerate -> right arrow
        }
        # mapping to pyautogui keys
        self.key_name = {'left': 'left', 'right': 'right'}

    def update_candidate(self, key, should_hold):
        """
        Call every frame with candidate state for a key.
        Commits the change only when the candidate persisted for frame_threshold frames.
        """
        if should_hold:
            self.counters[key] += 1
        else:
            self.counters[key] -= 1

        # clamp counters between -frame_threshold and +frame_threshold
        maxc = self.frame_threshold
        if self.counters[key] > maxc:
            self.counters[key] = maxc
        if self.counters[key] < -maxc:
            self.counters[key] = -maxc

        # If counter reaches +frame_threshold -> commit to hold
        if self.counters[key] == maxc and not self.committed[key]:
            self._hold_key(key)
            self.committed[key] = True

        # If counter reaches -frame_threshold -> commit to release
        if self.counters[key] == -maxc and self.committed[key]:
            self._release_key(key)
            self.committed[key] = False

    def _hold_key(self, key):
        # send keyDown only when not already held
        pyautogui.keyDown(self.key_name[key])

    def _release_key(self, key):
        pyautogui.keyUp(self.key_name[key])

    def release_all(self):
        # release anything currently held
        for k, held in self.committed.items():
            if held:
                self._release_key(k)
                self.committed[k] = False


def main(args):
    cam_index = args.cam
    min_det = args.min_det
    min_track = args.min_track
    frame_threshold = args.frames

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}")
        return

    controller = KeyController(frame_threshold=frame_threshold)

    # Keep last few FPS times
    fps_deque = deque(maxlen=8)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track
    ) as hands:
        try:
            while True:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Empty frame from camera.")
                    break

                # Flip for natural interaction (mirror)
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Convert color for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Default: no candidates for left/right this frame
                candidate_hold = {'left': False, 'right': False}

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label  # 'Left' or 'Right'
                        fingers = fingers_state(hand_landmarks, label)
                        num_open = sum(1 for f in fingers if f)

                        # define palm open vs fist
                        is_palm_open = (num_open >= 4)
                        is_fist = (num_open <= 1)

                        # Map to key candidates depending on handedness
                        # Left hand controls BRAKE (left arrow)
                        # Right hand controls ACCELERATE (right arrow)
                        if label.lower().startswith("left"):
                            candidate_hold['left'] = is_palm_open
                        elif label.lower().startswith("right"):
                            candidate_hold['right'] = is_palm_open

                        # draw landmarks and label
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        cx = int(hand_landmarks.landmark[0].x * w)
                        cy = int(hand_landmarks.landmark[0].y * h)
                        status_text = f"{label} Open:{is_palm_open} Fist:{is_fist} #open:{num_open}"
                        cv2.putText(frame, status_text, (cx - 80, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Update controller with candidates (debounced)
                controller.update_candidate('left', candidate_hold['left'])
                controller.update_candidate('right', candidate_hold['right'])

                # Draw HUD
                end = time.time()
                fps_deque.append(1.0 / max(1e-6, end - start))
                fps = sum(fps_deque) / len(fps_deque)
                left_state = "HELD" if controller.committed['left'] else "IDLE"
                right_state = "HELD" if controller.committed['right'] else "IDLE"
                cv2.putText(frame, f"FPS:{fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"Brake(left): {left_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,160,0), 2)
                cv2.putText(frame, f"Accel(right): {right_state}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,160,255), 2)

                # Show frame
                cv2.imshow("HCR Hand Control - Option B (Palm/Fist)", frame)

                # Exit keys: q or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        except KeyboardInterrupt:
            print("[INFO] Interrupted by user.")

        finally:
            # release resources and ensure keys are released
            controller.release_all()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hill Climb Racing - Hand control (Option B)")
    parser.add_argument("--cam", type=int, default=DEFAULT_CAM_INDEX, help="camera index (default 0)")
    parser.add_argument("--min_det", type=float, default=DEFAULT_MIN_DET, help="min detection confidence")
    parser.add_argument("--min_track", type=float, default=DEFAULT_MIN_TRACK, help="min tracking confidence")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_THRESHOLD, help="frames threshold for smoothing")
    args = parser.parse_args()
    main(args)
