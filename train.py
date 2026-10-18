import pandas as pd
import requests

print("Getting data from NASA...")

FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_srad"
]

columns = "kepid,koi_disposition," + ",".join(FEATURES)

query = f"select {columns} from cumulative"

url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=" + requests.utils.quote(query)
    + "&format=csv"
)

df = pd.read_csv(url)

print("Data downloaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns)

print("\nClass distribution:")
print(df["koi_disposition"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
# Keep only confirmed and false positive planets
df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]

# Remove rows with missing values
df = df.dropna()

print("\nAfter cleaning:")
print("Rows:", len(df))

print("\nClass distribution:")
print(df["koi_disposition"].value_counts())

# Prepare features and target

X = df[FEATURES]

y = df["koi_disposition"]

groups = df["kepid"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("Number of stars:", groups.nunique())

from sklearn.model_selection import GroupShuffleSplit

# Split data by star
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining classes:")
print(y_train.value_counts())

print("\nTesting classes:")
print(y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")
# Make predictions on test data
y_pred = model.predict(X_test)

print("\nPredictions completed!")
print("First 10 predictions:")
print(y_pred[:10])

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Feature importance

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Learning curve
training_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
f1_scores = []

for size in training_sizes:
    n = int(len(X_train) * size)

    temp_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    temp_model.fit(X_train.iloc[:n], y_train.iloc[:n])

    temp_pred = temp_model.predict(X_test)

    score = f1_score(
        y_test,
        temp_pred,
        pos_label="CONFIRMED"
    )

    f1_scores.append(score)

print("\nLearning Curve F1 Scores:")
for size, score in zip(training_sizes, f1_scores):
    print(f"{int(size * 100)}% training data: {score:.3f}")

plt.plot(
    [int(size * 100) for size in training_sizes],
    f1_scores,
    marker="o"
)

plt.xlabel("Training Data (%)")
plt.ylabel("F1 Score")
plt.title("Learning Curve")
plt.grid()
plt.show()

import joblib

joblib.dump(model, "model/random_forest_model.pkl")

print("Model saved successfully!")

