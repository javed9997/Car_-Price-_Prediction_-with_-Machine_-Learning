

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ---------------- Step 1: Load Dataset ----------------


try:
    df = pd.read_csv('car_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Creating a sample dataset for demo.")
    data = {
        'Car_Name': ['Swift', 'City', 'Corolla', 'Fortuner', 'i20', 'Baleno', 'Civic', 'Creta', 'Venue', 'Verna'],
        'Year': [2015, 2012, 2017, 2019, 2016, 2018, 2011, 2020, 2021, 2014],
        'Selling_Price': [3.5, 4.0, 8.0, 25.0, 5.5, 6.0, 7.0, 15.0, 12.0, 4.8],
        'Present_Price': [6.0, 10.0, 14.0, 35.0, 8.0, 9.0, 17.0, 20.0, 15.0, 8.5],
        'Driven_kms': [50000, 70000, 30000, 20000, 40000, 25000, 80000, 15000, 10000, 60000],
        'Fuel_Type': ['Petrol', 'Diesel', 'Diesel', 'Diesel', 'Petrol', 'Petrol', 'Diesel', 'Diesel', 'Petrol', 'Diesel'],
        'Seller_Type': ['Dealer', 'Individual', 'Dealer', 'Dealer', 'Dealer', 'Dealer', 'Individual', 'Dealer', 'Dealer', 'Individual'],
        'Transmission': ['Manual', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual'],
        'Owner': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)

# ---------------- Step 2: Data Preprocessing ----------------
print("\nFirst 5 rows of data:")
print(df.head())


label_encoders = {}
for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

# ---------------- Step 3: Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Step 4: Model Training ----------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- Step 5: Predictions ----------------
y_pred = model.predict(X_test)

# ---------------- Step 6: Evaluation ----------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ---------------- Step 7: Visualization ----------------
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.xlabel("Actual Selling Price (Lakhs)")
plt.ylabel("Predicted Selling Price (Lakhs)")
plt.title("Actual vs Predicted Car Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.show()

# ---------------- Step 8: Predict for New Data ----------------
new_data = pd.DataFrame({
    'Year': [2022],
    'Present_Price': [18.0],
    'Driven_kms': [5000],
    'Fuel_Type': [label_encoders['Fuel_Type'].transform(['Petrol'])[0]],
    'Seller_Type': [label_encoders['Seller_Type'].transform(['Dealer'])[0]],
    'Transmission': [label_encoders['Transmission'].transform(['Automatic'])[0]],
    'Owner': [0]
})

predicted_price = model.predict(new_data)
print(f"\nPredicted Selling Price for new car: {predicted_price[0]:.2f} Lakhs")
