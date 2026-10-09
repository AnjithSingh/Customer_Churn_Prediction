import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import json

def build_and_train(filepath, out_dir):
    df = pd.read_excel(filepath)
    
    # Preprocess Total Charges
    df['Total Charges'] = pd.to_numeric(df['Total Charges'].replace(" ", np.nan))
    df['Total Charges'] = df['Total Charges'].fillna(0)
    
    # Feature Engineering (Phase 6)
    df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=[-1, 12, 24, 48, 100], labels=['0-12', '13-24', '25-48', '49+']).astype(str)
    
    # Target
    df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    
    # Exclude Features (Phase 7)
    # Excluded for Data Leakage: Churn Label, Churn Value, Churn Score, Churn Reason
    # Excluded for Identifiers/Location: CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude
    # Excluded CLTV for prediction because it may implicitly contain future info and is better used for prioritization
    drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
                 'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Score', 
                 'Churn Reason', 'CLTV', 'Churn']
    
    X = df.drop(columns=drop_cols)
    y = df['Churn']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Pipeline
    num_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    cat_features = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 
                    'Phone Service', 'Multiple Lines', 'Internet Service', 
                    'Online Security', 'Online Backup', 'Device Protection', 
                    'Tech Support', 'Streaming TV', 'Streaming Movies', 
                    'Contract', 'Paperless Billing', 'Payment Method', 'Tenure Group']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ])
    
    # Models
    models = {
        'Baseline': DummyClassifier(strategy='stratified', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Cross Validation (Training set)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')
        
        # Train
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        # Evaluate on Test
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        res = {
            'Model': name,
            'CV F1 Mean': cv_scores.mean(),
            'CV F1 Std': cv_scores.std(),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob) if len(np.unique(y_prob)) > 2 else 'N/A'
        }
        results.append(res)
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{out_dir}/model_comparison.csv", index=False)
    
    # Tuning Random Forest (Example)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    search = RandomizedSearchCV(rf_pipe, param_grid, n_iter=5, cv=3, scoring='recall', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    # Threshold Optimization
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_best)
    
    # Find threshold that balances precision and recall, or favors recall slightly for business reasons
    fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(fscores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred_optimal = (y_prob_best >= best_threshold).astype(int)
    
    # Final Model Metrics
    final_metrics = {
        'Best Params': search.best_params_,
        'Threshold': float(best_threshold),
        'Accuracy': accuracy_score(y_test, y_pred_optimal),
        'Precision': precision_score(y_test, y_pred_optimal),
        'Recall': recall_score(y_test, y_pred_optimal),
        'F1': f1_score(y_test, y_pred_optimal),
        'ROC-AUC': roc_auc_score(y_test, y_prob_best)
    }
    
    with open(f"{out_dir}/final_model_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    # Save the Final Model Pipeline
    joblib.dump(best_model, 'models/final_model.joblib')
    
    # Feature Importance
    cat_enc = best_model.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = cat_enc.get_feature_names_out(cat_features)
    all_feature_names = num_features + list(cat_feature_names)
    
    importances = best_model.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    feat_imp.to_csv(f"{out_dir}/feature_importance.csv", index=False)
    
    # Risk Segmentation on full data
    full_probs = best_model.predict_proba(X)[:, 1]
    df['Churn Probability'] = full_probs
    df['Predicted Churn'] = (full_probs >= best_threshold).astype(int)
    
    def get_risk_segment(prob):
        if prob > 0.7: return 'High Risk'
        elif prob > 0.4: return 'Medium Risk'
        else: return 'Low Risk'
        
    df['Risk Segment'] = df['Churn Probability'].apply(get_risk_segment)
    
    # Save processed data for the app
    df.to_csv('data/processed/customer_predictions.csv', index=False)

if __name__ == "__main__":
    build_and_train("data/raw/Telco_customer_churn.xlsx", "reports")
