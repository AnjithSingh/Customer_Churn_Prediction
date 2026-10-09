import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
import json

def evaluate_model(filepath, out_dir):
    # 1. Load Data (recreating the exact split to evaluate the model on the test set)
    df = pd.read_excel(filepath)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'].replace(" ", np.nan))
    df['Total Charges'] = df['Total Charges'].fillna(0)
    df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=[-1, 12, 24, 48, 100], labels=['0-12', '13-24', '25-48', '49+']).astype(str)
    df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    
    drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
                 'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score', 
                 'Churn Reason', 'CLTV', 'Churn']
    
    X = df.drop(columns=drop_cols)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Load Model
    model = joblib.load('models/final_model.joblib')
    
    # 3. Load Threshold
    with open('reports/final_model_metrics.json', 'r') as f:
        metrics = json.load(f)
    threshold = metrics.get('Threshold', 0.5)
    
    # 4. Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay (0)', 'Churn (1)'], yticklabels=['Stay (0)', 'Churn (1)'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png")
    plt.close()
    
    # 6. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png")
    plt.close()
    
    # 7. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_curve.png")
    plt.close()

if __name__ == "__main__":
    evaluate_model("data/raw/Telco_customer_churn.xlsx", "reports/figures")
