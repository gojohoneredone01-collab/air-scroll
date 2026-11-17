import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam & Screen Setup
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# Smoothing variables for zero-lag cursor movement (EMA filter)
curr_x, curr_y = 0, 0
prev_x, prev_y = 0, 0
smooth_factor = 0.25 

is_dragging = False

print("Pro AI Mouse Started! Index=Move | Thumb+Index=Click | Thumb+Middle=Drag | Thumb+Pinky=Zoom | Index+Pinky+Thumb = SHUTDOWN | Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    shutdown_triggered = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Fingertips & MCP joints for finger curl check
            # Index (Tip: 8, PIP: 6), Pinky (Tip: 20, PIP: 18), Thumb (Tip: 4)
            tx, ty = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            ix, iy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            mx, my = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)
            px, py = int(hand_landmarks.landmark[20].x * w), int(hand_landmarks.landmark[20].y * h)
            
            # Middle and Ring MCP/PIP joints to check if they are folded down
            middle_pip = hand_landmarks.landmark[10].y
            ring_pip = hand_landmarks.landmark[14].y
            middle_tip = hand_landmarks.landmark[12].y
            ring_tip = hand_landmarks.landmark[16].y

            # --- CUSTOM SHUTDOWN GESTURE ---
            # Index extended up, Pinky extended out, Middle and Ring folded down
            index_extended = iy < hand_landmarks.landmark[6].y * h
            pinky_extended = py < hand_landmarks.landmark[18].y * h
            middle_folded = middle_tip > middle_pip
            ring_folded = ring_tip > ring_pip

            if index_extended and pinky_extended and middle_folded and ring_folded:
                cv2.putText(frame, "SHUTDOWN GESTURE DETECTED!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Pro AI Virtual Mouse", frame)
                cv2.waitKey(1000)  # Pause for 1 second
                shutdown_triggered = True
                break

            # Map Index Finger coordinates across screen with padding
            pad = 70
            screen_x = np.interp(ix, (pad, w - pad), (0, screen_width))
            screen_y = np.interp(iy, (pad, h - pad), (0, screen_height))
            
            # Exponential Moving Average (EMA) for zero lag & high smoothness
            curr_x = prev_x + (screen_x - prev_x) * smooth_factor
            curr_y = prev_y + (screen_y - prev_y) * smooth_factor
            
            # Move Cursor smoothly
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            # Calculate Gesture Distances
            click_dist = math.hypot(ix - tx, iy - ty)
            drag_dist = math.hypot(mx - tx, my - ty)
            zoom_dist = math.hypot(px - tx, py - ty)
            
            # 1. Left Click (Thumb + Index pinch)
            if click_dist < 35:
                cv2.circle(frame, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                pyautogui.sleep(0.15)
                
            # 2. Window Dragging (Thumb + Middle pinch)
            elif drag_dist < 35:
                if not is_dragging:
                    is_dragging = True
                    pyautogui.mouseDown()
                    cv2.putText(frame, "DRAGGING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if is_dragging:
                    is_dragging = False
                    pyautogui.mouseUp()
                    
            # 3. Zoom / Unzoom Gestures (Thumb + Pinky spread or pinch)
            if zoom_dist < 40:
                pyautogui.hotkey('ctrl', '-')  # Zoom Out
                cv2.putText(frame, "ZOOM OUT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                pyautogui.sleep(0.1)
            elif zoom_dist > 150:
                pyautogui.hotkey('ctrl', '+')  # Zoom In
                cv2.putText(frame, "ZOOM IN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                pyautogui.sleep(0.1)

    if shutdown_triggered:
        print("Shutdown gesture triggered successfully!")
        break

    cv2.imshow("Pro AI Virtual Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()