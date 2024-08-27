
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

def estimate_pi(N):
    inside_circle = 0

    x_inside = []
    y_inside = []
    x_outside = []
    y_outside = []

    for _ in range(N):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = math.sqrt(x**2 + y**2)

        if distance <= 1:
            inside_circle += 1
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    pi_estimate = (inside_circle / N) * 4

    return pi_estimate, x_inside, y_inside, x_outside, y_outside

N = 4000
pi_estimate, x_inside, y_inside, x_outside, y_outside = estimate_pi(N)
actual_pi = math.pi
pi_difference = abs(actual_pi - pi_estimate)

print(f"Estimated π: {pi_estimate:.6f}")
print(f"Actual π: {actual_pi:.6f}")
print(f"Difference: {pi_difference:.6f}")

plt.figure(figsize=(8, 8))
plt.scatter(x_inside, y_inside, color='blue', marker='.')
plt.scatter(x_outside, y_outside, color='red', marker='.')
plt.title(f"Monte Carlo Estimation of π (N = {N})")
plt.axis('equal')
plt.show()

def monte_carlo_pi(num_points):
    points_inside_circle = 0
    pi_values = []
    for i in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = math.sqrt(x**2 + y**2)
        if distance <= 1:
            points_inside_circle += 1
        pi_estimate = 4 * points_inside_circle / (i+1)
        pi_values.append(pi_estimate)
    return pi_values

num_points = 10000
pi_values = monte_carlo_pi(num_points)

# Set Seaborn style
sns.set(style="whitegrid")

# Plotting the estimated and actual values of pi
plt.figure(figsize=(10, 5))
sns.lineplot(data=pi_values, label='Estimated value of pi', color='blue', linewidth=2)
plt.axhline(y=math.pi, color='red', linestyle='-', label='Actual value of pi')
plt.title('Estimated and actual values of pi', fontsize=16)
plt.xlabel('Number of points', fontsize=14)
plt.ylabel('Value of pi', fontsize=14)
plt.legend(fontsize=12)
plt.show()

def calculate_difference_pi_estimate(n):
    pi_estimate, _, _, _, _ = estimate_pi(n)
    return abs(pi_estimate - math.pi)

def calculate_average_difference_pi_estimate(n, num_trials=100):
    differences = [calculate_difference_pi_estimate(n) for _ in range(num_trials)]
    return np.mean(differences)

data = [calculate_average_difference_pi_estimate(n) for n in [10**i for i in range(0, 6)]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plotting the data on the first subplot (log scale)
ax1.plot([10**i for i in range(0, 6)], data, marker='o', linestyle='-', label='Log Scale')
ax1.set_xscale('log')  
ax1.set_yscale('log')  
ax1.set_ylabel('Error')
ax1.set_title('Logarithmic Scale')
ax1.grid(True)
ax1.legend()

# Plotting the data on the second subplot (linear scale)
ax2.plot([10**i for i in range(0, 6)], data, marker='o', linestyle='-', color='orange', label='Linear Scale')
ax2.set_ylabel('Error')
ax2.set_title('Linear Scale')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

plt.show()