import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'area': [1000, 1500, 2000, 2500],
    'price': [300000, 450000, 600000, 750000]
}

df = pd.DataFrame(data)

X = df[['area']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Prediction for 1800 sq ft:", model.predict([[1800]]))
