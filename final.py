# Designed by Anirudh Rajesh
# **NOT FOR COMMERCIAL USE**

import csv
import math
import random
import time
import os
from collections import Counter

COLORS = ["\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[95m"]
RESET = "\033[0m"

def colorize(val):
    return COLORS[val % len(COLORS)] + str(val) + RESET

def line():
    print("━" * 60)

def loading(msg):
    print(msg, end="")
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print()

def intro():
    print("\n")
    line()
    print("    K-Nearest Neighbour Algorithm by Anirudh Rajesh    ")
    print("    © 2026 Anirudh Rajesh - All Rights Reserved    ")
    line()

    print("\nWhat this tool does:")
    print("→ Builds a Machine Learning model from scratch")
    print("→ No libraries like NumPy or pandas used")

    print("\n Pipeline:")
    print("1) Load dataset")
    print("2) Encode text → numbers")
    print("3) Normalize features")
    print("4) Train KNN model")
    print("5) Reduce dimensions using PCA")
    print("6) Visualize in terminal")
    print("7) Predict new data interactively")

    print("\n KNN Idea:")
    print("→ 'Tell me your neighbors, I'll tell who you are.'")

    line()

def file_explorer(start_path="."):
    print("\n TERMINAL FILE EXPLORER")
    print("Type number to select | '..' to go back | 'exit' to quit\n")

    current = os.path.abspath(start_path)

    while True:
        print(f"\n {current}\n")

        items = os.listdir(current)

        dirs = [d for d in items if os.path.isdir(os.path.join(current, d))]
        files = [f for f in items if os.path.isfile(os.path.join(current, f))]

        # show folders
        for i, d in enumerate(dirs):
            print(f"[{i}]  {d}")

        # show files
        for j, f in enumerate(files):
            print(f"[{j + len(dirs)}]  {f}")

        choice = input("\nSelect: ").strip()

        if choice == "exit":
            return None

        if choice == "..":
            current = os.path.dirname(current)
            continue

        if not choice.isdigit():
            print(" Invalid input")
            continue

        idx = int(choice)

        if idx < len(dirs):
            current = os.path.join(current, dirs[idx])
        else:
            file_idx = idx - len(dirs)
            if file_idx < len(files):
                selected = os.path.join(current, files[file_idx])
                print(f"\n Selected: {selected}")
                return selected
            else:
                print(" Invalid selection")

def load_csv(path):
    loading("\nLoading dataset")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        data = list(reader)

    header = data[0]
    rows = data[1:]

    print("Columns:", header)
    target = input("Enter target column: ")
    target_idx = header.index(target)

    X, y = [], []

    for row in rows:
        y.append(row[target_idx])
        X.append([row[i] for i in range(len(row)) if i != target_idx])

    print("\nSample data:")
    for i in range(min(3, len(X))):
        print(X[i], "→", y[i])

    return X, y

def encode_features(X):
    print("\nEncoding features...")
    cols = list(zip(*X))
    encoded = []
    mappings = []

    for i, col in enumerate(cols):
        try:
            new_col = [float(v) for v in col]
            print(f"Feature {i}: numeric")
            mappings.append(None)
        except:
            unique = list(set(col))
            mapping = {v: idx for idx, v in enumerate(unique)}
            print(f"Feature {i}: categorical → {mapping}")
            new_col = [mapping[v] for v in col]
            mappings.append(mapping)

        encoded.append(new_col)

    return list(zip(*encoded)), mappings

def encode_labels(y):
    print("\n Encoding labels...")
    unique = list(set(y))
    mapping = {v: i for i, v in enumerate(unique)}
    print("Label mapping:", mapping)

    return [mapping[v] for v in y]

def standardize(X):
    print("\nNormalizing features...")
    cols = list(zip(*X))

    means = [sum(col)/len(col) for col in cols]
    stds = []

    for i, (col, mean) in enumerate(zip(cols, means)):
        var = sum((x - mean)**2 for x in col)/len(col)
        std = math.sqrt(var) if var != 0 else 1
        stds.append(std)

        print(f"Feature {i}: mean={round(mean,2)}, std={round(std,2)}")

    X_scaled = []
    for row in X:
        X_scaled.append([(x-m)/s for x, m, s in zip(row, means, stds)])

    return X_scaled, means, stds

def train_test_split(X, y):
    print("\n Splitting dataset (80/20)...")
    data = list(zip(X, y))
    random.shuffle(data)

    split = int(len(data)*0.8)
    train = data[:split]
    test = data[split:]

    print(f"Train: {len(train)} | Test: {len(test)}")

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return list(X_train), list(X_test), list(y_train), list(y_test)

def euclidean(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def predict_one(X_train, y_train, x, k, debug=False):
    distances = [(euclidean(xi, x), yi) for xi, yi in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])

    if debug:
        print("\n Distance preview:")
        for d, label in distances[:5]:
            print(f"{round(d,3)} → class {label}")

    k = min(k, len(distances))
    neighbors = [label for _, label in distances[:k]]

    if debug:
        print("Nearest:", neighbors)

    return Counter(neighbors).most_common(1)[0][0]

def predict(X_train, y_train, X_test, k):
    preds = []
    for i, x in enumerate(X_test):
        preds.append(predict_one(X_train, y_train, x, k, debug=(i == 0)))
    return preds

def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b)/len(y_true)

def pca_2d(X):
    print("\n Running PCA (dimensionality reduction)...")

    n = len(X)
    d = len(X[0])

    means = [sum(row[i] for row in X)/n for i in range(d)]
    Xc = [[row[i]-means[i] for i in range(d)] for row in X]

    print("Means:", [round(m,2) for m in means])

    cov = [[sum(Xc[k][i]*Xc[k][j] for k in range(n))/n for j in range(d)] for i in range(d)]

    def power(mat):
        v = [random.random() for _ in range(len(mat))]
        for _ in range(50):
            v = [sum(mat[i][j]*v[j] for j in range(len(v))) for i in range(len(v))]
            norm = math.sqrt(sum(x*x for x in v)) or 1
            v = [x/norm for x in v]
        return v

    v1 = power(cov)

    def outer(v):
        return [[v[i]*v[j] for j in range(len(v))] for i in range(len(v))]

    lam = sum(v1[i]*sum(cov[i][j]*v1[j] for j in range(d)) for i in range(d))
    cov2 = [[cov[i][j]-lam*outer(v1)[i][j] for j in range(d)] for i in range(d)]

    v2 = power(cov2)

    print("Top directions found!")

    X2d = []
    for row in Xc:
        X2d.append([
            sum(row[i]*v1[i] for i in range(d)),
            sum(row[i]*v2[i] for i in range(d))
        ])

    return X2d

def ascii_plot(X, y, width=40, height=20):
    print("\n SCATTER PLOT")

    xs, ys = [p[0] for p in X], [p[1] for p in X]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    grid = [["." for _ in range(width)] for _ in range(height)]

    for (x, yv), label in zip(X, y):
        gx = int((x-min_x)/(max_x-min_x+1e-9)*(width-1))
        gy = int((yv-min_y)/(max_y-min_y+1e-9)*(height-1))
        gy = height-1-gy
        grid[gy][gx] = colorize(label)

    for row in grid:
        print(" ".join(row))

def ascii_boundary(X, y, k, width=40, height=20):

    print("\n")
    line()
    print(" DECISION BOUNDARY (what the model thinks)")
    line()
    
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    print("\nEach color = predicted class region\n")

    for i in range(height):
        row = []
        for j in range(width):
            # map grid → feature space
            x = min_x + (j / width) * (max_x - min_x)
            yv = max_y - (i / height) * (max_y - min_y)

            pred = predict_one(X, y, [x, yv], k)
            row.append(colorize(pred))

        print(" ".join(row))
        
def feature_importance(X, y, k):
    print("\n Feature importance (impact on accuracy):")

    base = accuracy(y, predict(X, y, X, k))

    for i in range(len(X[0])):
        X_mod = [row[:] for row in X]
        for row in X_mod:
            row[i] = 0

        acc = accuracy(y, predict(X_mod, y, X_mod, k))
        print(f"Feature {i}: {round(base - acc, 4)}")

def interactive(X_train, y_train, means, stds, mappings, k):
    print("\n INTERACTIVE MODE (type 'exit')")

    while True:
        inp = input(">> ")
        if inp == "exit":
            break

        vals = inp.split(",")
        row = []

        for i, v in enumerate(vals):
            try:
                row.append(float(v))
            except:
                row.append(mappings[i].get(v, 0) if mappings[i] else 0)

        row = [(x-m)/s for x, m, s in zip(row, means, stds)]

        pred = predict_one(X_train, y_train, row, k, debug=True)
        print("Prediction:", pred)

def main():
    intro()

    path = file_explorer()

    if not path:
        print("No file selected. Exiting.")
        return

    X, y = load_csv(path)

    X, mappings = encode_features(X)
    y = encode_labels(y)

    X, means, stds = standardize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    k = int(input("\nEnter k (neighbors): "))

    loading("Training KNN model")

    preds = predict(X_train, y_train, X_test, k)

    print("\nPredictions:", preds)
    print("Actual:     ", y_test)

    acc = accuracy(y_test, preds)
    print(f"\n Accuracy: {round(acc*100,2)}%")

    X_vis = pca_2d(X_train) if len(X_train[0]) > 2 else X_train

    ascii_plot(X_vis, y_train)
    ascii_boundary(X_vis, y_train, k)

    feature_importance(X_train, y_train, k)

    interactive(X_train, y_train, means, stds, mappings, k)


if __name__ == "__main__":
    main()
