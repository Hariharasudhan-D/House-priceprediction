import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
data = pd.read_csv("house.csv")

# # Display first 5 rows
# print("First 5 Rows:")
# print(data.head())

# # Display dataset information
# print("\nDataset Information:")
# print(data.info())

# # Display column names
# print("\nColumn Names:")
# print(data.columns)

# # Display dataset shape
# print("\nRows and Columns:")
# print(data.shape)

# # Check missing values
# print("Missing Values:")
# print(data.isnull().sum())


x = data[['Area', 'Rooms']]
y = data['Price']

# Split data to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train and evaluate model on train/test split
eval_model = LinearRegression()
eval_model.fit(X_train, y_train)
y_pred = eval_model.predict(X_test)
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Train the final model on the ENTIRE dataset for deployment/saving
# Using x.values to avoid "X does not have valid feature names" warnings when predicting with lists
model = LinearRegression()
model.fit(x.values, y.values)

# Predict house price for 1500 sq.ft and 3 rooms
predicted_price = model.predict([[1500, 3]])
print("Predicted price of 1500 sq.ft with 3 rooms is:")
print(predicted_price[0])

# Save the final model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model Saved Successfully")