
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\\Saniyah\\codingg\\tf-env\\House_pred\\Housing.csv")

df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})

X = df[['area', 'bedrooms', 'mainroad']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()


svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)


y_pred_scaled = svr.predict(X_test_scaled)


y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 6))


plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price', alpha=0.6)


plt.scatter(X_test['area'], y_pred, color='red', label='Predicted Price', alpha=0.6)

plt.xlabel('Area (sqft)')
plt.ylabel('Price')
plt.title('SVR Predictions vs Actual Prices (by Area)')
plt.legend()
plt.show()
