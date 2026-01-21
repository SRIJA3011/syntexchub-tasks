import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#  Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

print("Dataset Head:")
print(df.head())

#  Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())
sns.countplot(x="species_name", data=df)
plt.title("Class Distribution")
plt.show()

#  Feature Pair Visualization
sns.pairplot(df, hue="species_name")
plt.show()

#  Train-Test Split

X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#  Train Models

# Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


#  Model Evaluation Function

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    return acc

# Evaluate Models
acc_logreg = evaluate_model(y_test, y_pred_logreg, "Logistic Regression")
acc_dt = evaluate_model(y_test, y_pred_dt, "Decision Tree")


#  Accuracy Comparison

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [acc_logreg, acc_dt]
})

print("\nModel Accuracy Comparison:")
print(comparison_df)


# Interpretation of Misclassifications

print("""
Misclassification Interpretation:
- Most errors occur between Versicolor and Virginica.
- Setosa is almost perfectly classified due to clear feature separation.
- Overlapping petal measurements cause confusion between Versicolor and Virginica.
""")


#  CLI Prediction Script
def predict_species():
    print("\nEnter flower measurements:")
    sl = float(input("Sepal Length (cm): "))
    sw = float(input("Sepal Width (cm): "))
    pl = float(input("Petal Length (cm): "))
    pw = float(input("Petal Width (cm): "))

    sample = np.array([[sl, sw, pl, pw]])
    sample_scaled = scaler.transform(sample)

    prediction = logreg.predict(sample_scaled)[0]
    print(f"\nPredicted Species: {iris.target_names[prediction]}")

