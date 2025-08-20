import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


def plot_graph(x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_axis, y_axis, label="Persons", color="Purple")
    ax.set_title("Years of experience vs Salary")
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)

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