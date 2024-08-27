
import random

def simulate_shared_birthday(n, num_simulations):
    shared_birthday_count = 0
    for _ in range(num_simulations):
        birthdays = [random.randint(1, 365) for _ in range(n)]
        if len(birthdays) != len(set(birthdays)):
            shared_birthday_count += 1
    return shared_birthday_count / num_simulations

num_simulations = 10000
n = 2  # Start with n = 2
while True:
    probability = simulate_shared_birthday(n, num_simulations)
    if probability > 0.9:
        break
    n += 1

print("The smallest value of n for which P(B) exceeds 0.9 is:", n)

import random

def simulate_three_shared_birthdays(num_trials):
    num_people = 3
    while True:
        count_three_shared = 0
        for _ in range(num_trials):
            # Initialize a list to keep track of the number of people having a birthday on each day
            birthdays = [0] * 365
            for _ in range(num_people):
                # Generate a random birthday and increment the count for that day
                day = random.randint(0, 364)
                birthdays[day] += 1
            # Check if there is any day where at least three people have a birthday
            if max(birthdays) >= 3:
                count_three_shared += 1
        # Calculate the probability
        probability = count_three_shared / num_trials
        if probability > 0.5:
            break
        num_people += 1
    return num_people

num_trials = 10000
print("The smallest group size for which the probability of at least three people sharing a birthday exceeds 0.5 is approximately:", simulate_three_shared_birthdays(num_trials))