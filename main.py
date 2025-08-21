from cProfile import label
from idlelib.debugobj import myrepr

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def plot_graph(x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_axis, y_axis, label="Persons", color="Purple")
    ax.set_title("Years of experience vs Salary")
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)


class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.weight = None
        self.bias = None
        self.learning_rate = learning_rate
        self.iterations = iterations

    def train(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        number_samples, number_features = X_train.shape
        self.weight = np.zeros(number_features)
        self.bias = 0
        print(self.weight.shape)

        for _ in range(self.iterations):
            y_pred = np.dot(X_train, self.weight) + self.bias

            dw = -2/number_samples * np.dot(X_train.T, (y_train - y_pred))
            db = -2/number_samples * np.sum(y_train - y_pred)

            self.weight = self.weight - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

        print(f"Value for w = {self.weight}")
        print(f"Value for b = {self.bias}")

    def predict(self, X):
        y_pred = (np.dot(X, self.weight) + self.bias)
        return y_pred





st.markdown("<h1 style='text-align:center'>Simple Linear Regression</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h3>Uploading file.</h3>", unsafe_allow_html=True)

csv_file = st.file_uploader(label="", type=["csv"])
df = None
X, y = None, None
if csv_file is not None:
    df = pd.read_csv(csv_file)

    # Create X (independent variable) and y (dependent variable).
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

plot_button = st.button(label="Plot graph",)
if plot_button:
    if df is not None:
        plot_graph(X, y)
    else:
        st.error("There is no data loaded, be sure a csv file was uploaded.")

st.markdown("<h3>Training Simple Linear Regression model.</h3>", unsafe_allow_html=True)
train_button = st.button(label="Train model")


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

my_slr = SimpleLinearRegression()
my_slr.train(X_train, y_train)

print(X_test)
print(y_test)

predict = my_slr.predict(X_test)
print(predict)


print(type(y_test))
print(y_test.shape)
print(type(predict))
print(predict.shape)

def mean_square_errorr(test, predictions):
    return np.mean((test - predictions)**2)

mse1 = mean_square_errorr(y_test, predict)
mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)

print(f"MSE OG: {mse1:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")


y_pred_line = my_slr.predict(X_train)


fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_train, y_train, color="blue", label="Actual Data")
ax.plot(X_train, y_pred_line, color="red")
ax.legend()
st.pyplot(fig)