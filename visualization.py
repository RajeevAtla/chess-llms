import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iterations = list(range(10))

mean_train_loss = [
    0.5341,
    0.4151,
    0.2957,
    0.2930,
    0.3121,
    0.3467,
    0.3240,
    0.3600,
    0.3425,
    0.3501,
]

training_positions = [
    234,
    171,
    96,
    262,
    224,
    150,
    167,
    169,
    136,
    129,
]

# Game results from each self-play iteration.
# 1-0 = White win
# 0-1 = Black win
# 1/2-1/2 = draw/tie
game_results = [
    {"1-0": 5, "0-1": 2, "1/2-1/2": 13},
    {"1-0": 3, "0-1": 3, "1/2-1/2": 14},
    {"1-0": 1, "0-1": 2, "1/2-1/2": 17},
    {"1-0": 2, "0-1": 4, "1/2-1/2": 14},
    {"1-0": 1, "0-1": 6, "1/2-1/2": 13},
    {"1-0": 3, "0-1": 2, "1/2-1/2": 15},
    {"1-0": 4, "0-1": 2, "1/2-1/2": 14},
    {"1-0": 2, "0-1": 3, "1/2-1/2": 15},
    {"1-0": 4, "0-1": 0, "1/2-1/2": 16},
    {"1-0": 3, "0-1": 2, "1/2-1/2": 15},
]

df = pd.DataFrame({
    "iteration": iterations,
    "mean_train_loss": mean_train_loss,
    "training_positions": training_positions,
    "white_wins": [r.get("1-0", 0) for r in game_results],
    "black_wins": [r.get("0-1", 0) for r in game_results],
    "ties": [r.get("1/2-1/2", 0) for r in game_results],
})

df["total_games"] = df["white_wins"] + df["black_wins"] + df["ties"]
df["white_win_rate"] = df["white_wins"] / df["total_games"]
df["black_win_rate"] = df["black_wins"] / df["total_games"]
df["tie_rate"] = df["ties"] / df["total_games"]

display(df)

# ============================================================
# Plot 1: Mean training loss curve
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(
    df["iteration"],
    df["mean_train_loss"],
    marker="o",
    linewidth=2,
)

plt.title("Qwen2.5-0.5B Chess Self-Play Fine-Tuning Loss")
plt.xlabel("Self-Play Iteration")
plt.ylabel("Mean Training Loss")
plt.xticks(df["iteration"])
plt.grid(True, alpha=0.3)

for x, y in zip(df["iteration"], df["mean_train_loss"]):
    plt.text(x, y + 0.008, f"{y:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.show()

# ============================================================
# Plot 2: Game results as stacked bar chart
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    df["iteration"],
    df["white_wins"],
    label="White wins",
)

plt.bar(
    df["iteration"],
    df["ties"],
    bottom=df["white_wins"],
    label="Ties / Draws",
)

plt.bar(
    df["iteration"],
    df["black_wins"],
    bottom=df["white_wins"] + df["ties"],
    label="Black wins",
)

plt.title("Self-Play Game Results per Iteration")
plt.xlabel("Self-Play Iteration")
plt.ylabel("Number of Games")
plt.xticks(df["iteration"])
plt.yticks(range(0, int(df["total_games"].max()) + 2, 2))
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("game_results_stacked.png", dpi=200)
plt.show()

# ============================================================
# Plot 3: Game result rates
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["iteration"],
    df["white_win_rate"],
    marker="o",
    linewidth=2,
    label="White win rate",
)

plt.plot(
    df["iteration"],
    df["tie_rate"],
    marker="o",
    linewidth=2,
    label="Tie rate",
)

plt.plot(
    df["iteration"],
    df["black_win_rate"],
    marker="o",
    linewidth=2,
    label="Black win rate",
)

plt.title("Self-Play Game Result Rates")
plt.xlabel("Self-Play Iteration")
plt.ylabel("Rate")
plt.xticks(df["iteration"])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("game_result_rates.png", dpi=200)
plt.show()

# ============================================================
# Plot 4: Training positions generated per iteration
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["iteration"],
    df["training_positions"],
)

plt.title("Training Positions Kept per Self-Play Iteration")
plt.xlabel("Self-Play Iteration")
plt.ylabel("Training Positions")
plt.xticks(df["iteration"])
plt.grid(axis="y", alpha=0.3)

for x, y in zip(df["iteration"], df["training_positions"]):
    plt.text(x, y + 5, str(y), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("training_positions.png", dpi=200)
plt.show()

# ============================================================
# Summary stats
# ============================================================

total_white_wins = df["white_wins"].sum()
total_black_wins = df["black_wins"].sum()
total_ties = df["ties"].sum()
total_games = df["total_games"].sum()

print("Summary")
print("-------")
print(f"Total games: {total_games}")
print(f"White wins: {total_white_wins} ({total_white_wins / total_games:.1%})")
print(f"Black wins: {total_black_wins} ({total_black_wins / total_games:.1%})")
print(f"Ties/draws: {total_ties} ({total_ties / total_games:.1%})")
print(f"Best loss: {df['mean_train_loss'].min():.4f} at iteration {df.loc[df['mean_train_loss'].idxmin(), 'iteration']}")
print(f"Final loss: {df['mean_train_loss'].iloc[-1]:.4f}")
print(f"Total training positions: {df['training_positions'].sum()}")
