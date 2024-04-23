import pandas as pd
import numpy as np
import sympy as sy
from fancyimpute import IterativeImputer as MICE
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

print("NAME: VAISHVI JARIWALA; ID: 114433202\n")
# Load the datasets
df_iv = pd.read_csv('433202_IV.csv')
df_dv = pd.read_csv('433202_DV.csv')
df_partB = pd.read_csv('433202_partB.csv')

# Merge the IV and DV datasets based on 'ID'
merged_data = pd.merge(df_iv, df_dv, on='ID', how='outer')

# Placeholder data to simulate the structure of the Part B CSV file.
# You will replace this with the actual CSV reading code.
df_partB = pd.DataFrame({
    'x': np.random.rand(100) * 100,  # Random data to simulate 'x'
    'y': np.random.rand(100) * 50 + 50  # Random data to simulate 'y'
})

# Simulate log transformation if required by the assignment.
# This would be applied if the data is right-skewed or to linearize exponential relationships.
df_partB['x_log'] = np.log(df_partB['x'])
df_partB['y_log'] = np.log(df_partB['y'])

# Example of binning: Divide the range of 'x' into bins and label them.
num_bins = 10  # You'll need to determine the appropriate number of bins for your dataset.
bin_labels = [f'bin_{i}' for i in range(1, num_bins + 1)]
df_partB['x_binned'] = pd.cut(df_partB['x'], bins=num_bins, labels=bin_labels)

df_partB.head()  # Show the first few rows of the transformed/binned data.

# Count missing data for each column in the merged dataset
missing_data_counts = merged_data.isnull().sum()

# Display the count of missing values for each variable
print("Missing Data Counts before Imputation:")
print(missing_data_counts)

# Plot the IVs vs DVs Pre-Imputation (original data)
plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
plt.scatter(merged_data['IV'], merged_data['DV'], alpha=0.5)
plt.title('Original IVs vs DVs')
plt.xlabel('Independent Variables (IVs)')
plt.ylabel('Dependent Variables (DVs)')


# Impute missing data using MICE
mice_imputer = MICE()
merged_data_imputed = mice_imputer.fit_transform(merged_data)
merged_data_imputed = pd.DataFrame(merged_data_imputed, columns=merged_data.columns)

# Linear regression analysis for Part A
# Adding a constant for the intercept
X = sm.add_constant(merged_data_imputed['IV'])
y = merged_data_imputed['DV']
model_partA = sm.OLS(y, X).fit()
results_partA = model_partA.summary()

# Part B Analysis
# Assuming you need to perform similar steps for Part B
# If transformations or binning is required, you would need to add that here

# Adding a constant for the intercept
X_partB = sm.add_constant(df_partB['x'])
y_partB = df_partB['y']
model_partB = sm.OLS(y_partB, X_partB).fit()
results_partB = model_partB.summary()

# ANOVA for Part A
model_partA = ols('DV ~ IV', data=merged_data_imputed).fit()
anova_results_partA = sm.stats.anova_lm(model_partA, typ=2)
print("ANOVA Table for Part A:")
print(anova_results_partA)

# ANOVA for Part B (assuming you have a similar dataframe for Part B after any transformations)
model_partB = ols('y ~ x', data=df_partB).fit()  # replace 'df_partB_transformed' with your actual dataframe name
anova_results_partB = sm.stats.anova_lm(model_partB, typ=2)
print("\nANOVA Table for Part B:")
print(anova_results_partB)

# Plot IVs vs DVs Post-Imputation
plt.subplot(1, 3, 2)
plt.scatter(merged_data_imputed['IV'], merged_data_imputed['DV'], alpha=0.5, color='orange')
plt.title('Imputed IVs vs DVs')
plt.xlabel('Imputed Independent Variables (IVs)')
plt.ylabel('Imputed Dependent Variables (DVs)')

plt.subplot(1, 3, 3)
plt.scatter(df_partB['x'], df_partB['y'], alpha=0.5, label='Original')
plt.scatter(df_partB['x_log'], df_partB['y_log'], alpha=0.5, color='green', label='Log Transformed')
plt.title('Part B: Original and Transformed')
plt.xlabel('Independent Variable (x)')
plt.ylabel('Dependent Variable (y)')
plt.legend()

plt.tight_layout()
plt.show()

# Placeholder independent variable (IV) data
df_iv = pd.DataFrame({'IV': np.random.rand(100)})

# Placeholder dependent variable (DV) data
df_dv = pd.DataFrame({'DV': np.random.rand(100)})

# Calculate the correlation between IV and DV
correlation_iv_dv = df_iv['IV'].corr(df_dv['DV'])
correlation_iv_dv

correlation_iv_dv = df_iv['IV'].corr(df_dv['DV'])
print("Correlation between IV and DV:", correlation_iv_dv)


# Output results to console
print("Part A Results:")
print(results_partA)

print("\nPart B Results:")
print(results_partB)

# Save the summary to a text file
with open('output_partA.txt', 'w') as fh:
    fh.write(results_partA.as_text())

with open('output_partB.txt', 'w') as fh:
    fh.write(results_partB.as_text())