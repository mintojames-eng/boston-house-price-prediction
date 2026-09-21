# ============================================================
# HOUSE PRICE PREDICTION USING MULTIPLE LINEAR REGRESSION
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Boston House Price Prediction")
st.subheader("Multiple Linear Regression")

st.write(
    """
    This application predicts the median value of owner-occupied
    homes (MEDV) using Multiple Linear Regression.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Model Development",
        "Model Evaluation",
        "Visualizations",
        "New House Prediction"
    ]
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "boston_housing_clean.xlsx - Sheet1.csv"
    )

    return df


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found. Make sure "
        "'boston_housing_clean.xlsx - Sheet1.csv' "
        "is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

X = df.drop("MEDV", axis=1)

y = df["MEDV"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_test_pred
)

mse = mean_squared_error(
    y_test,
    y_test_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_test_pred
)


# Adjusted R²

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) *
    (n - 1) /
    (n - p - 1)
)


# Training R²

train_r2 = r2_score(
    y_train,
    y_train_pred
)


# ============================================================
# COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": model.coef_

})

coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    # Dataset dimensions

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Number of Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Number of Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Target Variable",
            "MEDV"
        )


    st.subheader("First 5 Records")

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    st.subheader("Last 5 Records")

    st.dataframe(
        df.tail(),
        use_container_width=True
    )


    st.subheader("Attribute Names")

    st.write(
        df.columns.tolist()
    )


    st.subheader("Data Types")

    st.dataframe(
        df.dtypes.astype(str).to_frame(
            "Data Type"
        ),
        use_container_width=True
    )


    # Missing values

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({

        "Column": missing_values.index,

        "Missing Values": missing_values.values

    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


    # Duplicates

    st.subheader("Duplicate Records")

    duplicate_count = df.duplicated().sum()

    if duplicate_count == 0:

        st.success(
            "No duplicate records found."
        )

    else:

        st.warning(
            f"{duplicate_count} duplicate records found."
        )


    # Statistics

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "Exploratory Data Analysis":

    st.header("📈 Exploratory Data Analysis")


    # --------------------------------------------------------
    # MEDV DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Distribution of House Prices (MEDV)"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.histplot(
        df["MEDV"],
        kde=True,
        ax=ax
    )

    ax.set_xlabel("MEDV")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Distribution of House Prices"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # SELECTED FEATURES
    # --------------------------------------------------------

    st.subheader(
        "Selected Features vs MEDV"
    )

    feature = st.selectbox(
        "Select Feature",
        ["RM", "LSTAT", "PTRATIO"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.scatterplot(
        data=df,
        x=feature,
        y="MEDV",
        ax=ax
    )

    ax.set_title(
        f"{feature} vs MEDV"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader(
        "Correlation Matrix"
    )

    correlation = df.corr(
        numeric_only=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # CORRELATION WITH MEDV
    # --------------------------------------------------------

    st.subheader(
        "Correlation with MEDV"
    )

    medv_corr = (
        correlation["MEDV"]
        .sort_values(
            ascending=False
        )
    )

    st.dataframe(
        medv_corr.to_frame(
            "Correlation"
        ),
        use_container_width=True
    )


# ============================================================
# MODEL DEVELOPMENT
# ============================================================

elif page == "Model Development":

    st.header(
        "🤖 Multiple Linear Regression"
    )


    st.subheader(
        "Train-Test Split"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training Samples",
            X_train.shape[0]
        )

    with col2:

        st.metric(
            "Testing Samples",
            X_test.shape[0]
        )


    st.subheader(
        "Regression Equation"
    )

    st.write(
        "The model uses multiple predictor variables "
        "to predict MEDV."
    )


    st.subheader(
        "Model Intercept"
    )

    st.write(
        model.intercept_
    )


    st.subheader(
        "Regression Coefficients"
    )

    st.dataframe(
        coefficients,
        use_container_width=True
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header(
        "📏 Model Evaluation"
    )


    # Metrics

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.4f}"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse:.4f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )

    with col4:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    with col5:

        st.metric(
            "Adjusted R²",
            f"{adjusted_r2:.4f}"
        )


    # Training vs Testing

    st.subheader(
        "Training vs Testing Performance"
    )

    performance_df = pd.DataFrame({

        "Dataset": [
            "Training",
            "Testing"
        ],

        "R² Score": [
            train_r2,
            r2
        ]

    })

    st.dataframe(
        performance_df,
        use_container_width=True
    )


    # Interpretation

    st.subheader(
        "R² Interpretation"
    )

    st.write(
        f"""
        The R² score of **{r2:.4f}** indicates the proportion
        of variation in MEDV explained by the regression model
        on the test dataset.
        """
    )


    # Overfitting / Underfitting

    difference = abs(
        train_r2 - r2
    )

    st.subheader(
        "Model Analysis"
    )

    if train_r2 > r2 and difference > 0.15:

        st.warning(
            "There is a relatively large difference between "
            "training and testing R², which may indicate "
            "overfitting."
        )

    elif train_r2 < 0.5 and r2 < 0.5:

        st.warning(
            "Both training and testing R² are relatively low, "
            "which may indicate underfitting."
        )

    else:

        st.success(
            "Training and testing performance are relatively "
            "close, with no strong indication of severe "
            "overfitting from this comparison."
        )


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "Visualizations":

    st.header(
        "📉 Model Visualizations"
    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Predicted MEDV"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_test,
        y_test_pred
    )

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    ax.set_xlabel(
        "Actual MEDV"
    )

    ax.set_ylabel(
        "Predicted MEDV"
    )

    ax.set_title(
        "Actual vs Predicted MEDV"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # RESIDUAL PLOT
    # --------------------------------------------------------

    st.subheader(
        "Residual Plot"
    )

    residuals = (
        y_test - y_test_pred
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_test_pred,
        residuals
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Predicted MEDV"
    )

    ax.set_ylabel(
        "Residual"
    )

    ax.set_title(
        "Residual Plot"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # COEFFICIENT PLOT
    # --------------------------------------------------------

    st.subheader(
        "Regression Coefficients"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        coefficients["Feature"],
        coefficients["Coefficient"]
    )

    ax.axvline(
        x=0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Regression Coefficient"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Regression Coefficients"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # POSITIVE / NEGATIVE COEFFICIENTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Largest Positive Coefficients"
        )

        positive = coefficients.sort_values(
            by="Coefficient",
            ascending=False
        )

        st.dataframe(
            positive[
                ["Feature", "Coefficient"]
            ].head(3),
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Largest Negative Coefficients"
        )

        negative = coefficients.sort_values(
            by="Coefficient",
            ascending=True
        )

        st.dataframe(
            negative[
                ["Feature", "Coefficient"]
            ].head(3),
            use_container_width=True
        )


# ============================================================
# NEW HOUSE PREDICTION
# ============================================================

elif page == "New House Prediction":

    st.header(
        "🔮 Predict MEDV for a New House"
    )

    st.write(
        """
        Enter values for the housing attributes below.
        The trained Multiple Linear Regression model will
        predict the approximate MEDV.
        """
    )


    # Create columns

    input_values = {}


    cols = st.columns(2)


    for i, feature in enumerate(X.columns):

        with cols[i % 2]:

            input_values[feature] = st.number_input(
                feature,
                value=float(
                    df[feature].median()
                ),
                step=0.1
            )


    st.divider()


    # Prediction button

    if st.button(
        "🔮 Predict House Price",
        type="primary"
    ):

        new_house = pd.DataFrame(
            [input_values]
        )

        prediction = model.predict(
            new_house
        )[0]


        st.success(
            f"Predicted MEDV: {prediction:.2f}"
        )


        st.info(
            """
            This is a model-based prediction and should
            not be interpreted as an actual market valuation.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "MAI511-2 Advanced Machine Learning\n\n"
    "Lab Exercise 1\n\n"
    "Multiple Linear Regression"
)