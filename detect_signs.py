from collections import deque
import math
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Chin tracking
chin_touch_time = None
chin_touch_y = None
thankyou_cooldown = 0  # Prevent spamming

# MediaPipe Hands
mp_hands = mp.solutions.hands

# Detection box
BOX_W, BOX_H = 700, 700

# Chatbox width
CHATBOX_W = 300

# Gesture history
gesture_history = deque(maxlen=10)


def landmarks_to_features(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
    return np.array(features)


def predict_sign(model, landmarks):
    features = landmarks_to_features(landmarks).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]


def is_j_motion(pinky_history):
    if len(pinky_history) < 10:
        return False

    points = np.array(pinky_history)
    deltas = np.diff(points, axis=0)
    
    angles = []
    for i in range(len(deltas) - 1):
        v1 = deltas[i]
        v2 = deltas[i + 1]
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot)
        angles.append(angle)

    total_angle = np.sum(angles)
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]

    return dx < -20 and dy > 20 and total_angle > 1.0


def main():
    global chin_touch_time, chin_touch_y, thankyou_cooldown

    # Load trained model
    try:
        with open('trained_asl_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("❌ trained_asl_model.pkl not found. Run train_model.py first.")
        return

    cap = cv2.VideoCapture(0)
    pinky_history = deque(maxlen=10)
    j_detected_time = None

    last_prediction = None
    prediction_start_time = None
    confirmed_sign = ""

    mp_face_mesh = mp.solutions.face_mesh

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
         ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            box_x = (frame_w - BOX_W) // 2
            box_y = (frame_h - BOX_H) // 2

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands and face
            results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            raw_prediction = None

            # ---- CHIN TRACKING ----
            chin_point = None
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                chin_lm = face_landmarks.landmark[152]
                chin_x = int(chin_lm.x * frame_w)
                chin_y = int(chin_lm.y * frame_h)
                chin_point = (chin_x, chin_y)
                cv2.circle(frame, chin_point, 5, (255, 0, 0), -1)

            if chin_point and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                middle_finger_tip = hand_landmarks.landmark[12]
                palm_x = int(middle_finger_tip.x * frame_w)
                palm_y = int(middle_finger_tip.y * frame_h)

                cv2.circle(frame, (palm_x, palm_y), 10, (0, 255, 0), -1)

                dist = math.hypot(palm_x - chin_point[0], palm_y - chin_point[1])
                current_time = time.time()

                if dist < 60:
                    chin_touch_time = current_time
                    chin_touch_y = palm_y

                elif chin_touch_time and (current_time - chin_touch_time < 1.0):
                    if palm_y - chin_touch_y > 40 and (current_time - thankyou_cooldown > 2):
                        confirmed_sign = "Thank you"
                        thankyou_cooldown = current_time
                        chin_touch_time = None
                        chin_touch_y = None

            # ---- HAND PROCESSING ----
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                pinky_tip = hand_landmarks.landmark[20]
                pinky_x = int(pinky_tip.x * frame_w)
                pinky_y = int(pinky_tip.y * frame_h)
                pinky_history.append((pinky_x, pinky_y))

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_tips = [(int(hand_landmarks.landmark[i].x * frame_w),
                                int(hand_landmarks.landmark[i].y * frame_h)) for i in [4, 8, 12, 16, 20]]

                inside_box = all(box_x <= x <= box_x + BOX_W and box_y <= y <= box_y + BOX_H
                                 for (x, y) in finger_tips)

                if inside_box:
                    predicted = predict_sign(model, hand_landmarks.landmark)

                    if predicted == 'I' and is_j_motion(pinky_history):
                        raw_prediction = "J"
                        j_detected_time = time.time()
                    else:
                        raw_prediction = predicted

                    current_time = time.time()
                    if raw_prediction == last_prediction:
                        if prediction_start_time is None:
                            prediction_start_time = current_time
                        elif current_time - prediction_start_time >= 0.5:
                            confirmed_sign = raw_prediction
                            if not gesture_history or gesture_history[-1] != confirmed_sign:
                                gesture_history.append(confirmed_sign)
                    else:
                        prediction_start_time = current_time
                        last_prediction = raw_prediction

            # --- DRAWING BOX ---
            box_color = (0, 255, 0) if confirmed_sign else (0, 0, 255)
            cv2.rectangle(frame, (box_x, box_y),
                          (box_x + BOX_W, box_y + BOX_H), box_color, 3)

            # --- CHATBOX ---
            canvas = np.zeros((frame_h, frame_w + CHATBOX_W, 3), dtype=np.uint8)
            canvas[:, :frame_w, :] = frame
            canvas[:, frame_w:, :] = (50, 50, 50)  # dark gray background

            cv2.putText(canvas, "Detected Gestures", (frame_w + 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            for i, gesture in enumerate(list(gesture_history)[::-1]):
                y = 80 + i * 40
                cv2.putText(canvas, f"{gesture}", (frame_w + 20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- CURRENT DETECTION ---
            if confirmed_sign:
                cv2.putText(canvas, f"Detected: {confirmed_sign}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

            if j_detected_time and (time.time() - j_detected_time < 3):
                cv2.putText(canvas, "J detected", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            elif j_detected_time:
                j_detected_time = None

            cv2.imshow("ASL Sign Detector", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
