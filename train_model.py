# train_model.py

import cv2
import mediapipe as mp
import csv
from collections import defaultdict

mp_hands = mp.solutions.hands

def landmarks_to_features(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
    return features

def count_existing_samples(csv_file='sign_data.csv'):
    counts = defaultdict(int)
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    counts[row[0]] += 1
    except FileNotFoundError:
        pass  # No samples yet
    return counts

def main():
    csv_file = 'sign_data.csv'
    cap = cv2.VideoCapture(0)
    counts = count_existing_samples(csv_file)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            print("🎥 Starting data collection.")
            print("🔤 Press A-Z to record a sample. Press Q to quit.\n")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = landmarks_to_features(hand_landmarks.landmark)

                cv2.putText(frame, "Press A-Z to save sample, Q to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Collect Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif ord('a') <= key <= ord('z'):
                    if results.multi_hand_landmarks:
                        label = chr(key).upper()
                        writer.writerow([label] + features)
                        counts[label] += 1
                        print(f"✅ Saved sample for '{label}' — Total: {counts[label]}")
                elif key == ord('1'):  # New key for "I love you"
                    if results.multi_hand_landmarks:
                        label = "I love you"
                        writer.writerow([label] + features)
                        counts[label] += 1
                        print(f"💖 Saved sample for '{label}' — Total: {counts[label]}")
                elif key == ord('2'):  # New key for "H2P"
                    if results.multi_hand_landmarks:
                        label = "H2P!!!!"
                        writer.writerow([label] + features)
                        counts[label] += 1
                        print(f"💖 Saved sample for '{label}' — Total: {counts[label]}")
                elif key == ord('3'):  # New key for "Hello"
                    if results.multi_hand_landmarks:
                        label = "Hello!!!!"
                        writer.writerow([label] + features)
                        counts[label] += 1
                        print(f"💖 Saved sample for '{label}' — Total: {counts[label]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
