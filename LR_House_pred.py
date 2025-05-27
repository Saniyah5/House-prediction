import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"C:\\Saniyah\\codingg\\tf-env\\House_pred\\Housing.csv")


df['mainroad_encoded'] = LabelEncoder().fit_transform(df['mainroad'])


X = df[['area', 'bedrooms', 'mainroad_encoded']]
y = df['price']


model = LinearRegression()
model.fit(X, y)


area_range = np.linspace(df['area'].min(), df['area'].max(), 100)


fixed_bedrooms = 2
fixed_mainroad = 1  # 1 = Yes


X_plot = pd.DataFrame({
    'area': area_range,
    'bedrooms': fixed_bedrooms,
    'mainroad_encoded': fixed_mainroad
})


y_pred = model.predict(X_plot)


plt.figure(figsize=(10, 6))
plt.scatter(df['area'], y, color='gray', label='Actual Prices')
plt.plot(area_range, y_pred, color='blue', linewidth=2, label='Regression Line\n(bedrooms=2, mainroad=yes)')

plt.xlabel('Area (sqft)')
plt.ylabel('Price')
plt.title('Linear Regression: Area vs Price (Fixed Bedrooms & Mainroad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
