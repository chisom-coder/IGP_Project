import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm

files = ['assessments.csv', 'courses.csv', 'studentAssessment.csv',
         'studentInfo.csv', 'studentRegistration.csv', 'studentVle.csv', 'vle.csv']

merged_df = pd.read_csv(files[0])  # Read the first file

for file in files[1:]:
    df = pd.read_csv(file)
    merged_df = merged_df.merge(df, how='outer')

# Optional: Reset the index of the merged DataFrame
merged_df.reset_index(drop=True, inplace=True)


#merged_df.to_csv('merged_data.csv', index=False)

merged_df.shape

merged_df = merged_df.drop(['num_of_prev_attempts', 'is_banked','date','date_submitted','date_registration','date_submitted','id_site','activity_type','week_from','week_to','module_presentation_length','code_module','code_presentation','date_unregistration'],axis=1)

merged_df.shape

#merged_df.to_csv('filter.csv',index=False)

merged_df.isna().sum()

merged_df.dropna(inplace=True)

merged_df.shape

merged_df.to_csv('final_filter.csv',index=False)


descriptive_stats = merged_df[['sum_click', 'score']].describe()
print(descriptive_stats)


merged_df[['sum_click', 'score']].hist(bins=10, figsize=(10, 6))
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution Comparison')
plt.legend(['sum_click', 'score'])
plt.show()


# Extract the 'sum_click' and 'score' columns
sum_click = merged_df['sum_click']
score = merged_df['score']

# Calculate the correlation coefficient
correlation_coefficient, p_value = pearsonr(sum_click, score)
print("Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)

# Create a scatter plot
plt.scatter(sum_click, score)
plt.xlabel('sum_click')
plt.ylabel('score')
plt.title('Relationship between sum_click and score')
plt.show()

# Perform a t-test
t_statistic, p_value_ttest = ttest_ind(sum_click, score)
print("T-test Statistic:", t_statistic)
print("P-value (T-test):", p_value_ttest)




# Create the design matrix
X = merged_df['sum_click']  # Independent variable
y = merged_df['score']  # Dependent variable

# Add a constant term to the design matrix
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the regression results
print(results.summary())




# Identify the number of rows with '10/20/2023' in 'imd_band' column
count = merged_df[merged_df['imd_band'] == '20-Oct'].shape[0]
print("Number of rows with '20-Oct' in 'imd_band':", count)


# Drop the rows with '10/20/2023' in 'imd_band' column
merged_df = merged_df[merged_df['imd_band'] != '20-Oct']


# Drop rows containing '10/20/2023' anywhere in the dataset
merged_df = merged_df[~merged_df.astype(str).eq('10/20/2023').any(axis=1)]

merged_df.shape

# Display unique values in 'imd_band' column
imd_band_values = merged_df['imd_band'].unique()
print("Unique values in 'imd_band' column:")
for value in imd_band_values:
    print(value)




# Select the column you want to view distinct values from

#column_name = 'imd_band'



# Get distinct values in the selected column

#distinct_values = merged_df[column_name].unique()



# Print the distinct values

for value in distinct_values:

  print(value)
  
  
  