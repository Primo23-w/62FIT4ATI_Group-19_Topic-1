Diabetes Risk Prediction Using MLP
This repository implements a Feedforward Neural Network (MLP) for predicting diabetes risk from CDC BRFSS data. The model handles class imbalance with SMOTE and focuses on high Recall for medical relevance.
Features

Binary classification: Healthy (0) vs. At-Risk (1).
21 input features (demographics, health, lifestyle).
Optimized for imbalance (~84% healthy).

Repository Contents

diabetes_risk_prediction.ipynb: Jupyter notebook with code, explanations, and results.
diabetes_risk_model.joblib: Trained MLP model.
scaler.joblib: StandardScaler for input preprocessing.
data.csv: Dataset (download from CDC BRFSS if not included).
requirements.txt: Dependencies.

Guide to Setup
Follow these steps to set up the project locally:

Clone the Repository:textgit clone https://github.com/yourusername/diabetes-risk-prediction-mlp.git
cd diabetes-risk-prediction-mlp
Create a Virtual Environment (recommended to isolate dependencies):textpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
Ensure Python 3.8+ is installed. Then run:textpip install -r requirements.txtIf requirements.txt is not present, install manually:textpip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
Download Dataset (if not included in the repo):
Obtain 'data.csv' from the CDC BRFSS survey data (e.g., via Kaggle or official CDC site).
Place it in the repository root.

Launch Jupyter:textjupyter notebookOpen diabetes_risk_prediction.ipynb in your browser.

Guide to Reproduction
To reproduce the results exactly as in the notebook:

Prepare the Environment:
Follow the setup guide above to clone, activate the virtual environment, install dependencies, and ensure 'data.csv' is available.

Run the Notebook:
Start Jupyter: jupyter notebook.
Open diabetes_risk_prediction.ipynb.
Run all cells sequentially (Kernel > Restart & Run All).
This will:
Load and preprocess the data (cleaning, scaling, SMOTE for imbalance).
Train the MLP model (using scikit-learn's MLPClassifier with Adam optimizer and early stopping).
Evaluate performance (focus on Recall, Precision-Recall curve).
Perform inference on test samples and custom profiles.
Export the trained model as diabetes_risk_model.joblib and scaler as scaler.joblib (if not already present).



Verify Outputs:
Check console outputs for class distribution, visualizations (e.g., imbalance plot, BMI histogram).
Confirm metrics: Expect high Recall for the minority class due to optimizations.
Test inference:Pythonfrom joblib import load
import numpy as np
import pandas as pd

# Load model and scaler
model = load('diabetes_risk_model.joblib')
scaler = load('scaler.joblib')

# Example custom input (21 features matching the dataset columns)
# Columns: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
custom_input = np.array([[1, 1, 1, 35, 1, 0, 1, 0, 0, 1, 0, 1, 0, 4, 15, 10, 1, 1, 10, 4, 2]])
custom_df = pd.DataFrame(custom_input, columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'])
scaled_input = scaler.transform(custom_df)
prediction, prob = model.predict(scaled_input), model.predict_proba(scaled_input)[:, 1]
print(f"Prediction: {prediction[0]} (1 = At Risk), Probability: {prob[0]:.4f}")
Adjust the threshold (e.g., 0.3) as per the notebook for custom predictions.

Troubleshooting Reproduction Issues:
Dependency Versions: If results differ, check requirements.txt for exact versions (e.g., scikit-learn==1.5.0).
Random Seeds: The notebook uses random_state=42 for reproducibility; ensure no changes.
Data Integrity: Verify 'data.csv' matches the expected shape (269131, 22) and no missing values after cleaning.
Hardware/Env Differences: Results may vary slightly due to floating-point precision; focus on trends (e.g., Recall > 0.8).

Training from Scratch (Optional):
If you want to retrain without using the pre-trained model, comment out loading steps and run the training cells.
Training time: ~Minutes on a standard CPU (depends on dataset size).