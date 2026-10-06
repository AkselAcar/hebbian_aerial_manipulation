import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f_best = np.load("results/f_best.npy")
fitness = -f_best  # CMA-ES stores it negated (it minimizes); flip back to actual reward

plt.figure(figsize=(8, 5))
plt.plot(fitness, marker="o")
plt.xlabel("Generation")
plt.ylabel("Best fitness (reward)")
plt.title("Hebbian + CMA-ES: best fitness per generation")
plt.grid(True)
plt.savefig("results/fitness_plot.png", dpi=150)
print("Saved results/fitness_plot.png")