import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Past sales data
past_months = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # Months
past_sales = np.array([100, 150, 200, 250, 300, 350])     # Sales

# Linear regression model
model = LinearRegression()
model.fit(past_months, past_sales)

# Predicting future sales for the next 3 months
future_months = np.array([7, 8, 9]).reshape(-1, 1)  # Months
future_sales = model.predict(future_months)

# Plotting the results
plt.scatter(past_months, past_sales, color='blue', label='Past Sales')
plt.plot(future_months, future_sales, color='red', linestyle='--', label='Predicted Sales')
plt.title('Sales Prediction')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Printing predicted sales for future months
for month, sales in zip(future_months, future_sales):
    print(f"Predicted sales for month {int(month)}: ${sales:.2f}")
