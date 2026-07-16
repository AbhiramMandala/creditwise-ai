import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import preprocess_data

# Unified function to calculate performance metrics
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = [1 if y == "good" else 0 for y in y_pred]
    
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="good"),
        "recall": recall_score(y_test, y_pred, pos_label="good"),
        "f1": f1_score(y_test, y_pred, pos_label="good"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

@st.cache_resource
def train_models(data):
    data = preprocess_data(data)
    X = data.drop("class", axis=1)
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train all 6 models
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_metrics = evaluate_model(knn_model, X_test, y_test, "K-Nearest Neighbors")

    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_metrics = evaluate_model(svm_model, X_test, y_test, "Support Vector Machine")

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

    return (
        rf_model, lr_model, gb_model, knn_model, svm_model, dt_model,
        X.columns,
        rf_metrics, lr_metrics, gb_metrics, knn_metrics, svm_metrics, dt_metrics
    )

def credit_risk_prediction(data):
    st.header("Credit Risk Prediction")
    st.write("Enter customer details to check their credit risk.")

    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.number_input("Loan Duration (months)", min_value=0, value=12)
    checking_status = st.selectbox("Checking Account Status", ["<0", "0<=X<200", ">=200", "no checking"])
    credit_history = st.selectbox("Credit History", ["critical/other existing credit", "existing paid", "delayed previously", "no credits/all paid", "all paid"])
    purpose = st.selectbox("Loan Purpose", ["radio/tv", "education", "furniture/equipment", "new car", "used car", "business", "other"])
    savings_status = st.selectbox("Savings Account Status", ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"])
    employment = st.selectbox("Employment Status", ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"])
    personal_status = st.selectbox("Personal Status", ["male single", "female div/dep/mar", "male mar/wid", "female single"])
    housing = st.selectbox("Housing Status", ["own", "rent", "for free"])
    job = st.selectbox("Job Type", ["unskilled resident", "skilled", "high qualif/self emp/mgmt", "unemp/unskilled non res"])

    if st.button("Check Credit Risk"):
        input_data = pd.DataFrame({
            "age": [age], "credit_amount": [credit_amount], "duration": [duration],
            "checking_status": [checking_status], "credit_history": [credit_history],
            "purpose": [purpose], "savings_status": [savings_status],
            "employment": [employment], "personal_status": [personal_status],
            "housing": [housing], "job": [job]
        })
        input_data = pd.get_dummies(input_data, drop_first=True)

        models = train_models(data)

        rf_model, lr_model, gb_model, knn_model, svm_model, dt_model, X_columns = models[:7]
        rf_metrics, lr_metrics, gb_metrics, knn_metrics, svm_metrics, dt_metrics = models[7:]

        for col in X_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[X_columns]

        predictions = {
            "Random Forest": rf_model.predict(input_data)[0],
            "Logistic Regression": lr_model.predict(input_data)[0],
            "Gradient Boosting": gb_model.predict(input_data)[0],
            "K-Nearest Neighbors": knn_model.predict(input_data)[0],
            "Support Vector Machine": svm_model.predict(input_data)[0],
            "Decision Tree": dt_model.predict(input_data)[0]
        }

        st.write("### Predictions from each model:")
        for model, pred in predictions.items():
            risk_label = "Low Risk ✅" if pred == "good" else "High Risk ❌"
            st.write(f"**{model}: {risk_label}**")

        st.write("---")
        st.write("### Model Performance Comparison")
        def display_metrics(metrics):
            st.write(f"**{metrics['model']} Metrics:**")
            st.write(f"- Accuracy: {metrics['accuracy']:.2f}")
            st.write(f"- Precision: {metrics['precision']:.2f}")
            st.write(f"- Recall: {metrics['recall']:.2f}")
            st.write(f"- F1 Score: {metrics['f1']:.2f}")
            st.write(f"- ROC AUC: {metrics['roc_auc']:.2f}")
            st.write("")

        for metrics in [rf_metrics, lr_metrics, gb_metrics, knn_metrics, svm_metrics, dt_metrics]:
            display_metrics(metrics)

        st.write("---")
        st.write("### Confusion Matrices")
        def plot_confusion_matrix(metrics, ax, title):
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["High Risk", "Low Risk"], yticklabels=["High Risk", "Low Risk"])
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        matrices = [rf_metrics, lr_metrics, gb_metrics, knn_metrics, svm_metrics, dt_metrics]
        titles = ["Random Forest", "Logistic Regression", "Gradient Boosting", "KNN", "SVM", "Decision Tree"]

        for i, (metrics, title) in enumerate(zip(matrices, titles)):
            plot_confusion_matrix(metrics, axes[i//3, i%3], title)

        st.pyplot(fig)
