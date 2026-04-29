A fully interactive **Machine Learning toolkit built from scratch using pure Python** — no external libraries like NumPy, pandas, or scikit-learn.

This project demonstrates how core ML concepts actually work under the hood, with **step-by-step explanations, visualizations, and real-time predictions**.

---

## What This Project Does

This tool takes a dataset and:

* Loads and processes raw data
* Converts categorical data into numbers
* Normalizes features
* Applies the **K-Nearest Neighbors (KNN)** algorithm
* Reduces dimensions using **PCA**
* Visualizes data using ASCII graphics
* Predicts outcomes for new inputs
* Shows feature importance
* Allows interactive predictions

---

## How It Works (Simple Explanation)

1. **Dataset Input**
   You provide a CSV dataset (e.g. customer data).

2. **Data Processing**
   Text values → converted into numbers
   Features → normalized for fair comparison

3. **KNN Algorithm**
   For a new data point:

   * Finds the *k closest neighbors*
   * Uses majority voting
   * Outputs the predicted class

4. **PCA (Dimensionality Reduction)**
   High-dimensional data → reduced to 2D
   Makes visualization possible

5. **Visualization**

   * ASCII scatter plots
   * Decision boundary maps
   * Colored clusters

6. **Interactive Mode**
   Enter your own data → get predictions instantly

---

## Example

Input:

```text
25,30000,15,Chennai,Android,Female
```

Output:

```text
Prediction: Buy
```

With explanation:

```text
Closest neighbors → [Buy, Buy, NoBuy]
Final decision → Buy
```

---

## Features

* KNN implemented from scratch
* PCA (with eigenvector computation)
* ASCII-based visualization
* Decision boundary mapping
* Feature importance scoring
* Interactive CLI predictions
* No external dependencies

---

##  Dataset Format

Example:

```csv
age,income,visits,city,device,gender,label
22,25000,12,Chennai,Android,Male,Buy
30,42000,18,Chennai,Android,Female,Buy
45,60000,6,Delhi,iOS,Female,NoBuy
```

* Last column → target (label)
* Others → features

---

##  How to Run

```bash
double-click on the executable according to your operating system
```

Then:

1. Select dataset
2. Choose target column
3. Enter value of `k`
4. View results and interact

---

##  Interactive Mode

After training:

```text
>> Enter values:
```

Example:

```text
>> 28,32000,10,Bangalore,Android,Male
```

Output:

```text
Prediction: NoBuy
```

---

##  Visual Output

* ASCII scatter plot (data distribution)
* Decision boundary (model behavior)
* Colored class regions

---

## Why This Project Matters

Most ML tools hide the logic behind libraries.

This project:

* No black boxes
* No shortcuts
* Full transparency
* Learn-by-seeing approach

It helps you truly understand:

* Distance calculations
* Feature scaling
* Model decision making

---

##  Limitations

* Not optimized for large datasets
* Slower than library-based implementations
* Designed for learning, not production

---

##  Future Improvements

* Confusion matrix
* Probability predictions
* Auto-tuning for best `k`
* Web interface (React + Python backend)

---

##  Author

Built by Anirudh Rajesh

---

##  Final Thought

> “Don’t just use machine learning — understand it.”

---
