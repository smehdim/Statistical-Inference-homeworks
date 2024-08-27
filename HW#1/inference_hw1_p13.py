
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def gamble(initial_stake, bet_amount, win_probability, target_amount, iterations):
    stake = initial_stake
    stake_history = [stake]
    iteration_count = []

    for i in range(iterations):
        if stake <= 0:
            break  # The gambler is broke
        if stake >= target_amount:
            break  # The gambler reached the target
        if np.random.rand() < win_probability:
            stake += bet_amount
        else:
            stake -= bet_amount
        stake_history.append(stake)
        iteration_count.append(i+1)

    return stake_history, iteration_count

def simulate_gambler(initial_stake, bet_amount, win_probability, target_amount, iterations, rounds):
    final_stakes = []
    iteration_counts = []

    for _ in range(rounds):
        stake_history, iteration_count = gamble(initial_stake, bet_amount, win_probability, target_amount, iterations)
        final_stakes.append(stake_history[-1])
        iteration_counts.append(iteration_count[-1])

    return final_stakes, iteration_counts

def main():
    initial_stake = 100
    bet_amount = 10
    win_probability = 0.5
    target_amount = 200
    iterations = 1000
    rounds = 1000

    final_stakes, iteration_counts = simulate_gambler(initial_stake, bet_amount, win_probability, target_amount, iterations, rounds)

    # Calculate the chance of winning
    win_count = sum(final_stake >= target_amount for final_stake in final_stakes)
    win_percentage = (win_count / rounds) * 100

    print(f"Chance of winning: {win_percentage:.2f}%")

    iteration_counts_series = pd.Series(iteration_counts)
    expanding_averages = iteration_counts_series.expanding().mean()

    sns.set_theme(style="whitegrid", palette="pastel")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(10, rounds)):  # Plot up to 10 rounds
        ax.plot(gamble(initial_stake, bet_amount, win_probability, target_amount, iterations)[0], label=f'Round {i+1}')
    ax.set_title('Gambler\'s Ruin Simulations for 10 rounds')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Stake')
    sns.despine()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(expanding_averages, marker='o', color='skyblue', markersize=1, label='Expanding Average',)
    ax.set_title('Average Number of Iterations Vs rounds')
    ax.set_xlabel('Round')
    ax.set_ylabel('Average Number of Iterations')
    sns.despine()
    plt.show()

main()