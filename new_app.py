# Import Library
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly as py



# Load the breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[data.feature_names], df['target'], test_size=0.2, random_state=42)

# Define the models and their hyperparameters
models = {
    'Logistic Regression': {'model': LogisticRegression, 'params': {}},
    'Decision Tree': {'model': DecisionTreeClassifier, 'params': {'max_depth': [3, 5, 7]}},
    'Random Forest': {'model': RandomForestClassifier, 'params': {'n_estimators': [50, 100, 200]}},
    'Gradient Boosting': {'model': GradientBoostingClassifier, 'params': {'learning_rate': [0.01, 0.1, 1.0]}},
    'Support Vector Machine': {'model': SVC, 'params': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}},
    'K-Nearest Neighbors': {'model': KNeighborsClassifier, 'params': {'n_neighbors': [3, 5, 7]}},
    'Gaussian Naive Bayes': {'model': GaussianNB, 'params': {}},
    'Multi-layer Perceptron': {'model': MLPClassifier, 'params': {'hidden_layer_sizes': [(100,), (50, 50), (20, 20, 20)]}},
    'Linear Discriminant Analysis': {'model': LinearDiscriminantAnalysis, 'params': {}},
    'AdaBoost': {'model': AdaBoostClassifier, 'params': {'n_estimators': [50, 100, 200]}},
    'Gaussian Process': {'model': GaussianProcessClassifier, 'params': {}},
    'XGBoost': {'model': XGBClassifier, 'params': {'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}}
}

# Create a sidebar widget for model selection
model_choice = st.sidebar.selectbox('Select a model', list(models.keys()))

# Get the selected model's hyperparameters
model_params = models[model_choice]['params']
model_param_values = {}

# Create a sidebar widget for each hyperparameter
for param, values in model_params.items():
    model_param_values[param] = st.sidebar.selectbox(param, values)

# Train and evaluate the selected model with the chosen hyperparameters
model = models[model_choice]['model'](**model_param_values)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Project title
st.title('Final Year Project:')
st.header("Machine Learning model Comparison for Breast Cancer Classification")

# Explain the breast cancer dataset
st.header('**Breast Cancer Dataset:**')
st.write('The Breast Cancer dataset contains features computed from breast mass images and corresponding diagnosis of either malignant (1) or benign (0).')
st.write('It includes various attributes such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.')
st.write('The goal is to classify whether a breast mass is malignant or benign based on these features.')

# Explain the selected model
st.header('**Selected Model:**', model_choice)
st.write('The', model_choice, 'model is a', model.__class__.__name__, 'which is a', model.__class__.__bases__[0].__name__)
st.write('It is a supervised learning model that can be used for classification tasks.')
st.write('The model uses the following hyperparameters:')
for param, value in model_param_values.items():
    st.write('-', param, ':', value)



# Display evaluation metrics
st.write('**Accuracy:**', accuracy)
st.write('**Precision:**', precision)
st.write('**Recall:**', recall)
st.write('**F1-Score:**', f1)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"fontsize": 14})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)

# Display classification report
report = classification_report(y_test, y_pred)
st.code(report, language='text')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
st.pyplot(fig)

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)
fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
st.pyplot(fig)

# Perform exploratory data analysis
st.subheader('Exploratory Data Analysis')

# Display summary statistics
st.write('**Summary Statistics**')
st.write(df.describe())


# Display correlation matrix
st.write('**Correlation Matrix**')
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', cbar=False, annot_kws={"fontsize": 8})
st.pyplot(plt)

# Interactive Correlation Matrix
st.write('**Interactive Correlation Matrix**')
fig = px.imshow(corr_matrix)
st.plotly_chart(fig)


