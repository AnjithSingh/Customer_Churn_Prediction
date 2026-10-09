import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

def run_eda(filepath, output_dir):
    df = pd.read_excel(filepath)
    
    # Clean Total Charges
    df['Total Charges'] = pd.to_numeric(df['Total Charges'].replace(" ", np.nan))
    # For tenure=0, Total Charges is NaN, let's fill with 0
    df['Total Charges'] = df['Total Charges'].fillna(0)
    
    # Target
    df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    
    # Categorical Analysis - Chi-Square
    cat_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 
                'Phone Service', 'Multiple Lines', 'Internet Service', 
                'Online Security', 'Online Backup', 'Device Protection', 
                'Tech Support', 'Streaming TV', 'Streaming Movies', 
                'Contract', 'Paperless Billing', 'Payment Method']
    
    stats_results = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df['Churn Label'])
        chi2, p, dof, expected = chi2_contingency(contingency)
        stats_results.append({
            'Variable': col,
            'Test Used': 'Chi-Square',
            'Test Statistic': chi2,
            'p-value': p,
            'Significant': p < 0.05
        })
        
        # Plot
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, hue='Churn Label')
        plt.title(f'Churn by {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/churn_by_{col.replace(' ', '_')}.png")
        plt.close()
        
    # Numerical Analysis - T-Test
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    for col in num_cols:
        churned = df[df['Churn'] == 1][col]
        retained = df[df['Churn'] == 0][col]
        t_stat, p = ttest_ind(churned.dropna(), retained.dropna(), equal_var=False)
        stats_results.append({
            'Variable': col,
            'Test Used': "Welch's t-test",
            'Test Statistic': t_stat,
            'p-value': p,
            'Significant': p < 0.05
        })
        
        # Plot
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x='Churn Label', y=col)
        plt.title(f'{col} by Churn Status')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/churn_by_{col.replace(' ', '_')}.png")
        plt.close()

    # Tenure groups
    df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=[-1, 12, 24, 48, 100], labels=['0-12', '13-24', '25-48', '49+'])
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='Tenure Group', hue='Churn Label')
    plt.title('Churn by Tenure Group')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_by_Tenure_Group.png")
    plt.close()

    # Save stats results
    pd.DataFrame(stats_results).to_csv(f"{output_dir}/statistical_tests.csv", index=False)
    
if __name__ == "__main__":
    run_eda("data/raw/Telco_customer_churn.xlsx", "reports/figures")
