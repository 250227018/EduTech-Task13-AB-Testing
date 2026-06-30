import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.api as sms

# 1. Load dataset
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/ab_data.csv")
print(df.head())
print(df.shape)
print(df['group'].value_counts())

# 2. Group A (control) aur Group B (treatment) separate karo
group_A = df[df['group'] == 'control']['converted']
group_B = df[df['group'] == 'treatment']['converted']

# 3. Conversion rates calculate karo
conv_rate_A = group_A.mean()
conv_rate_B = group_B.mean()
print(f"\nGroup A Conversion Rate: {conv_rate_A:.4f}")
print(f"Group B Conversion Rate: {conv_rate_B:.4f}")

# 4. Hypothesis Setup
print("\n--- Hypothesis ---")
print("H0 (Null): No difference between Group A and Group B conversion rate")
print("H1 (Alternative): There IS a difference between Group A and Group B")

# 5. T-test perform karo
t_stat, p_value = stats.ttest_ind(group_A, group_B)
print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# 6. Decision
#alpha = 0.05
#if p_value < alpha:
 #   print("Result: Reject Null Hypothesis - Significant difference found! ✓")
#else:
 #   print("Result: Fail to Reject Null Hypothesis - No significant difference ✗")

# 7. Confidence Interval (95%)
ci_A = sms.DescrStatsW(group_A).tconfint_mean()
ci_B = sms.DescrStatsW(group_B).tconfint_mean()
print(f"\n95% Confidence Interval - Group A: {ci_A}")
print(f"95% Confidence Interval - Group B: {ci_B}")

# 8. Visualization - Conversion Rate Comparison
plt.figure(figsize=(6, 4))
rates = pd.DataFrame({
    'Group': ['A (Control)', 'B (Treatment)'],
    'Conversion Rate': [conv_rate_A, conv_rate_B]
})
sns.barplot(x='Group', y='Conversion Rate', data=rates, palette='Set2')
plt.title('A/B Test: Conversion Rate Comparison')
plt.savefig('conversion_comparison.png')
plt.show()

# 9. Distribution Plot
plt.figure(figsize=(8, 5))
sns.histplot(group_A, color='blue', label='Group A', alpha=0.5, kde=True)
sns.histplot(group_B, color='orange', label='Group B', alpha=0.5, kde=True)
plt.title('Conversion Distribution - A vs B')
plt.legend()
plt.savefig('distribution_plot.png')
plt.show()

# 10. Report save karo
alpha= 0.05
report = pd.DataFrame({
    'Metric': ['Conversion Rate A', 'Conversion Rate B', 'T-statistic', 'P-value', 'Significant?'],
    'Value': [conv_rate_A, conv_rate_B, t_stat, p_value, p_value < alpha]
})
report.to_csv('ab_test_report.csv', index=False)
print("\nReport saved!")