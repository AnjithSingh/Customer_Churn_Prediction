# Customer Churn Prediction & Retention Analytics

## Project Overview
This project focuses on predicting customer churn for a telecommunications company and providing actionable business recommendations to improve customer retention. It features a complete end-to-end data science workflow, including exploratory data analysis, statistical testing, machine learning modeling, and an interactive Streamlit application for executive dashboards and prediction.

## Business Problem
The telecommunications company is experiencing customer churn, which negatively impacts revenue and growth. The business needs to identify which customers are most likely to leave, understand the driving factors behind their departure, and determine actionable strategies to retain high-risk segments.

## Objectives
1. **Which customers are most likely to churn?** Build a predictive classification model to identify high-risk customers.
2. **What characteristics are associated with customer churn?** Perform statistical and exploratory data analysis to find churn drivers.
3. **Which customer segments have particularly high churn risk?** Segment customers based on churn probability and value.
4. **What actions could the company take to improve customer retention?** Deliver data-backed business recommendations to reduce churn.

## Dataset
The project uses `Telco_customer_churn.xlsx`, containing over 7,043 customer records with 33 features.

## Dataset Description
Key features include:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Account Info**: Tenure Months, Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges
- **Services**: Phone Service, Multiple Lines, Internet Service, Online Security, Tech Support, Streaming TV, Streaming Movies

## Data Quality
- **Missing Values**: 11 instances of blank spaces in `Total Charges` were imputed with `0` as they represented customers with `0` tenure months.
- **Data Types**: `Total Charges` was converted from object to float.
- **Data Leakage**: `Churn Value`, `Churn Score`, and `Churn Reason` were explicitly removed from model training to prevent target leakage. Identifiers like `CustomerID`, and overly granular geographic data (`City`, `Zip Code`) were also dropped to ensure the model generalizes well. `CLTV` was reserved for post-prediction business prioritization.

## Exploratory Data Analysis
- Analyzed churn distribution by demographics, services, and account information using Chi-Square tests.
- Analyzed numerical relationships (Tenure, Charges) using Welch's t-test.

## Key Findings
- **Tenure**: Customers with lower tenure (0-12 months) have the highest churn rate.
- **Contract Type**: Month-to-month contracts are significantly more likely to churn than one-year or two-year contracts.
- **Services**: Lack of Online Security and Tech Support strongly correlates with higher churn.
- **Internet**: Fiber optic customers churn more frequently than DSL customers.

## Feature Engineering
- Created `Tenure Group` to bin continuous tenure into business-meaningful intervals.
- Handled categorical encoding via OneHotEncoder and continuous features via StandardScaler in a Scikit-Learn Pipeline.

## Machine Learning Models
Tested Models:
- Baseline (DummyClassifier)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Model Comparison
Evaluated models using 5-fold cross-validation on the training set (stratified). Metrics considered included Precision, Recall, F1-score, and ROC-AUC. 

## Final Model
**Random Forest Classifier** (tuned via RandomizedSearchCV). It provided the best balance of predictive power, robustness, and interpretability (via feature importance).

## Evaluation
Recall was prioritized as a business metric, since false negatives (failing to identify a churning customer) are generally more expensive than false positives (offering a retention discount to someone who would have stayed anyway). The classification threshold was optimized using the Precision-Recall curve to maximize the F1 score and capture more true positives.

## Explainability
Feature importance extracted from the Random Forest model highlighted:
1. `Contract_Month-to-month`
2. `Tenure Months`
3. `Total Charges` / `Monthly Charges`
4. `Internet Service_Fiber optic`

## Customer Risk Segmentation
Predicted probabilities are segmented into:
- **High Risk**: > 70% probability of churn
- **Medium Risk**: 40% - 70% probability
- **Low Risk**: < 40% probability

## Churn Reason Analysis
While not used for modeling (due to data leakage), exploratory analysis of `Churn Reason` indicates that "Competitor made better offer" and "Attitude of support person" are common post-churn reasons, supporting the model's findings that lack of tech support and pricing are key factors.

## Business Recommendations
1. **Contract Conversion**: Target month-to-month customers with incentives to upgrade to 1-year contracts.
2. **Onboarding**: Proactive 90-day onboarding for new customers.
3. **Service Bundling**: Include Tech Support and Online Security in standard packages.
4. **Prioritized Retention**: Focus outreach on High Risk + High CLTV customers.

## Streamlit Application
The application provides an interactive Executive Dashboard, a Customer Prediction tool, Churn Drivers visualizations, and Segmentation analysis.

## Project Structure
```text
customer-churn-retention/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
│   └── figures/
├── app/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
```bash
pip install -r requirements.txt
```

## Run Application
```bash
streamlit run app/app.py
```

## Future Improvements
- **Temporal Churn Modeling**: Incorporating time-series behavior (e.g., changes in usage over time).
- **Cost-Sensitive Learning**: Optimizing the model explicitly for the dollar cost of interventions.
- **Model Monitoring**: Setting up a pipeline to detect concept drift in production.

## Limitations
- **Observational Data**: Results show correlation, not necessarily causation.
- **Snapshot Data**: Lacks historical timeline of customer actions prior to churn.
- **Geographic Limits**: All data is from California, limiting generalizability to other states.
