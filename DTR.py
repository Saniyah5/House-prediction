import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

X = np.array([[1000], [1500], [2000], [2500], [3000]])

y = np.array([200000, 250000, 300000, 350000, 400000])

model = DecisionTreeRegressor()

model.fit(X, y)
predicted_price = model.predict([[2200]])
print(f"Predicted price for 2200 sqft house: ${predicted_price[0]:,.2f}")

X_test = np.arange(1000, 3001, 10).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, color='green', label='Training Data')

plt.plot(X_test, y_pred, color='yellow', label='Decision Tree Prediction')

plt.xlabel("House Size (sqft)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction with Decision Tree")
plt.legend()
plt.grid(True)
plt.show()

