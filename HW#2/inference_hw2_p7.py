
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

df = pd.read_excel('/content/drive/MyDrive/datasets/inference/hw2/Problem 7 Data.xls')

"""# p1"""

population_values = df['mortalities']

# Create a histogram using seaborn
sns.histplot(population_values, bins=15, kde=False, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('mortalities')
plt.ylabel('Frequency')
plt.title('Histogram of Population for Breast Cancer Mortality')

# Show the plot
plt.show()

"""# p2"""

population_values = df['population']
cancer_mortality_values = df['mortalities']

# Calculate mean and total cancer mortality
population_mean = population_values.mean()
total_cancer_mortality = cancer_mortality_values.sum()

# Calculate variance and standard deviation
population_variance = population_values.var(ddof=0)
population_std_dev = population_values.std(ddof=0)

print(f"Population Mean: {population_mean}")
print(f"Total Cancer Mortality: {total_cancer_mortality}")
print(f"Population Variance: {population_variance}")
print(f"Population Standard Deviation: {population_std_dev}")

"""# c

"""

cancer_mortality_values = df['mortalities']


sample_size = 25

# Number of samples to generate
num_samples = 1000

# Initialize an array to store sample means
sample_means = np.zeros(num_samples)

# Generate random samples and calculate means
for i in range(num_samples):
    sample = np.random.choice(cancer_mortality_values, size=sample_size, replace=True)
    sample_means[i] = np.mean(sample)

plt.hist(sample_means, bins=25, edgecolor='black')
plt.xlabel('Sample Mean of Cancer Mortality')
plt.ylabel('Frequency')
plt.title(f'Sampling Distribution of the Mean (Sample Size = {sample_size})')
plt.show()

"""# d"""

# Simple random sample of size 25
sample_size = 25
random_sample = np.random.choice(cancer_mortality_values, size=sample_size)

# Estimate the mean and total cancer mortality from the sample
sample_mean = np.mean(random_sample)
total_cancer_mortality_sample = sample_mean * len(df)

print(f"Estimated Sample Mean: {sample_mean}")
print(f"Estimated Total Cancer Mortality in Sample: {total_cancer_mortality_sample}")

sample_variance = np.var(random_sample, ddof=1)
sample_std_dev = np.std(random_sample, ddof=1)

# Print the results
print(f"Estimated Sample Variance: {sample_variance}")
print(f"Estimated Sample Standard Deviation: {sample_std_dev}")

"""# f

"""

dof = len(random_sample) - 1

# Confidence level
t_score = t.ppf(0.975, df=dof)

margin_of_error =  t_score * (sample_std_dev / np.sqrt(len(random_sample)))

mean_confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
total_confidence_interval = (mean_confidence_interval[0] * len(df),
                             mean_confidence_interval[1] * len(df))

# Print the results
print(f"95% Confidence Interval for Population Mean: {mean_confidence_interval}")
print(f"95% Confidence Interval for Population Total: {total_confidence_interval}")

# Check if the intervals cover the population values
population_mean = np.mean(cancer_mortality_values)
population_total = np.sum(cancer_mortality_values)

print(f"Does the mean interval cover the population mean? {mean_confidence_interval[0] <= population_mean <= mean_confidence_interval[1]}")
print(f"Does the total interval cover the population total? {total_confidence_interval[0] <= population_total <= total_confidence_interval[1]}")

"""# g"""

# Simple random sample of size 25
sample_size = 100
random_sample = np.random.choice(cancer_mortality_values, size=sample_size)

# Estimate the mean and total cancer mortality from the sample
sample_mean = np.mean(random_sample)
total_cancer_mortality_sample = np.sum(random_sample)

# Print the results
print(f"Estimated Sample Mean: {sample_mean}")
print(f"Estimated Total Cancer Mortality in Sample: {total_cancer_mortality_sample}")

sample_variance = np.var(random_sample, ddof=1)
sample_std_dev = np.std(random_sample, ddof=1)

# Print the results
print(f"Estimated Sample Variance: {sample_variance}")
print(f"Estimated Sample Standard Deviation: {sample_std_dev}")

dof = len(random_sample) - 1

# Confidence level
t_score = t.ppf(0.975, df=dof)

# Calculate the margin of error
margin_of_error =  t_score * (sample_std_dev / np.sqrt(len(random_sample)))

# Calculate the confidence intervals for the population mean and total
mean_confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
total_confidence_interval = (mean_confidence_interval[0] * len(df),
                             mean_confidence_interval[1] * len(df))

print(f"95% Confidence Interval for Population Mean: {mean_confidence_interval}")
print(f"95% Confidence Interval for Population Total: {total_confidence_interval}")

# Check if the intervals cover the population values
population_mean = np.mean(cancer_mortality_values)
population_total = np.sum(cancer_mortality_values)

print(f"Does the mean interval cover the population mean? {mean_confidence_interval[0] <= population_mean <= mean_confidence_interval[1]}")
print(f"Does the total interval cover the population total? {total_confidence_interval[0] <= population_total <= total_confidence_interval[1]}")

"""# i"""

cancer_mortality_values = df['mortalities']
total_population_values = df['population']

# Simple random sample of size 25
sample_size = 25
random_sample_cancer_mortality = np.random.choice(cancer_mortality_values, size=sample_size, replace=False)
random_sample_total_population = np.random.choice(total_population_values, size=sample_size, replace=False)

# Calculate the ratio estimator for the sample
sample_ratio_estimator = np.mean(random_sample_cancer_mortality) / np.mean(random_sample_total_population)

# Number of simulation runs
num_simulations = 1000

# Initialize an array to store ratio estimators from simulations
simulated_ratio_estimators = np.zeros(num_simulations)

# Simulate the sampling distribution
for i in range(num_simulations):
    # Draw a new random sample for each simulation
    sim_sample_cancer_mortality = np.random.choice(cancer_mortality_values, size=sample_size, replace=False)
    sim_sample_total_population = np.random.choice(total_population_values, size=sample_size, replace=False)

    # Calculate the ratio estimator for each simulation
    simulated_ratio_estimators[i] = np.mean(sim_sample_cancer_mortality) / np.mean(sim_sample_total_population)

# Plot the sampling distribution of ratio estimators
plt.hist(simulated_ratio_estimators, bins=25, edgecolor='black')
plt.xlabel('Ratio Estimator of Mean Cancer Mortality')
plt.ylabel('Frequency')
plt.title(f'Sampling Distribution of Ratio Estimators (Sample Size = {sample_size})')
plt.show()

# Compare with the result of part (c)
population_mean = np.mean(cancer_mortality_values)
print(f"Population Mean (Part c): {population_mean}")
print(f"Sample Ratio Estimator (Part c): {np.mean(random_sample_cancer_mortality)}")

# Calculate the mean and standard deviation of the simulated ratio estimators
mean_simulated_ratio_estimators = np.mean(simulated_ratio_estimators)
std_dev_simulated_ratio_estimators = np.std(simulated_ratio_estimators)

# Print the results
print(f"\nMean of Simulated Ratio Estimators: {mean_simulated_ratio_estimators}")
print(f"Standard Deviation of Simulated Ratio Estimators: {std_dev_simulated_ratio_estimators}")

"""# j"""

sample_size = 25
sample_indices = np.random.choice(len(cancer_mortality_values), size=sample_size, replace=False)
sample_cancer_mortality = cancer_mortality_values.iloc[sample_indices]
sample_total_population = total_population_values.iloc[sample_indices]

# Calculate ratio estimates for the sample
ratio_estimate_mean = np.mean(sample_cancer_mortality / sample_total_population) * np.mean(total_population_values)
ratio_estimate_total = np.sum(sample_cancer_mortality) / np.sum(sample_total_population) * np.sum(total_population_values)

# Print the results
print(f"Ratio Estimate of Population Mean: {ratio_estimate_mean}")
print(f"Ratio Estimate of Population Total: {ratio_estimate_total}")

random_sample = np.random.choice(cancer_mortality_values, size=sample_size, replace=False)
sample_mean = np.mean(random_sample)
total_cancer_mortality_sample = sample_mean * len(df)

print(f"\nEstimate of Population Mean (Part d): {sample_mean}")
print(f"Estimate of Population Total (Part d): {total_cancer_mortality_sample}")

sample_size = 25
sample_indices = np.random.choice(len(cancer_mortality_values), size=sample_size, replace=False)
sample_cancer_mortality = cancer_mortality_values.iloc[sample_indices]

# Calculate ratio estimates for the sample
ratio_estimate_mean = np.mean(sample_cancer_mortality / sample_total_population) * np.mean(total_population_values)

# Calculate standard errors for ratio estimates
ratio_se_mean = np.std(sample_cancer_mortality / sample_total_population, ddof=1) * np.mean(total_population_values) / np.sqrt(sample_size)

# Degrees of freedom for t-distribution
df = sample_size - 1

# Confidence level
confidence_level = 0.95

# Calculate the margin of error
margin_of_error_mean = t.ppf((1 + confidence_level) / 2, df) * ratio_se_mean

# Calculate the confidence intervals for the ratio estimates
ratio_mean_confidence_interval = (ratio_estimate_mean - margin_of_error_mean, ratio_estimate_mean + margin_of_error_mean)

# Print the results
print(f"95% Confidence Interval for Ratio Estimate of Population Mean: {ratio_mean_confidence_interval}")
print(f"95% Confidence Interval for Ratio Estimate of Population Total: {ratio_total_confidence_interval}")