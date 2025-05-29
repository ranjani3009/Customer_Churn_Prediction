import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


df = pd.read_csv(r"C:\Users\ranja\OneDrive\Desktop\churn-bigml-80.csv")

print("Initial data shape:", df.shape)
print("Churn unique values before mapping:", df['Churn'].unique())

df.drop(columns=['State', 'Area code', 'Phone'], inplace=True, errors='ignore')


churn_map = {}


unique_churn = df['Churn'].unique()

for val in unique_churn:
    if str(val).strip().lower() in ['true.', 'true', 'yes', '1']:
        churn_map[val] = 1
    elif str(val).strip().lower() in ['false.', 'false', 'no', '0']:
        churn_map[val] = 0
    else:
        churn_map[val] = np.nan  

df['Churn'] = df['Churn'].map(churn_map)

print("Churn mapping applied:", churn_map)
print("Churn value counts after mapping:\n", df['Churn'].value_counts(dropna=False))


df.dropna(subset=['Churn'], inplace=True)


df['Churn'] = df['Churn'].astype(int)


df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())


target = df['Churn']
features = df.drop('Churn', axis=1)


numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
features_numeric = features[numeric_features]

print("Data shape after preprocessing:", df.shape)
print("Numeric feature columns:", list(numeric_features))
print("Features numeric shape:", features_numeric.shape)
print("Target distribution:\n", target.value_counts(dropna=False))


if features_numeric.shape[0] == 0:
    raise ValueError("No samples available after preprocessing. Check your dataset and preprocessing steps.")


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)


X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print(f"\nBest Random Forest Parameters: {grid_search.best_params_}")
print(f"Best Random Forest F1 Score: {f1_score(y_test, y_pred_best):.4f}")


conf_matrix = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Best Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()