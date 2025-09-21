# import_csv

import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter

def load_data(filename='sign_data.csv'):
    labels = []
    features = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[0])
            features.append([float(x) for x in row[1:]])
    return np.array(features), np.array(labels)

def train():
    X, y = load_data()

    print(f"📊 Dataset Size: {len(y)} samples")
    print(f"📚 Label Distribution: {Counter(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    print(f"✅ Training Accuracy: {model.score(X_train, y_train) * 100:.2f}%")
    print(f"🧪 Test Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    with open('trained_asl_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print("💾 Model saved as 'trained_asl_model.pkl'")

if __name__ == "__main__":
    train()