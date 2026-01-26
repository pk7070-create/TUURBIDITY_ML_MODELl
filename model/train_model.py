import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/turbidity_data.csv")
print(df)

X = df[['turbidity']]   # input
y = df['label']         # output

print("X (input):")
print(X.head())

print("y (output):")
print(y.head())



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Decision Tree trained successfully")


y_pred = model.predict(X_test)

print("Predictions on test data:")
print(y_pred)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)




# 1️⃣ Model train
model.fit(X_train, y_train)

# 2️⃣ Training vs Testing accuracy (ONE TIME)
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))


while True:
    val = int(input("Enter turbidity (-1 to exit): "))
    if val == -1:
        break
    print("Water Quality:", model.predict([[val]])[0])






