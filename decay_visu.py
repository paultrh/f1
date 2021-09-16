import matplotlib.pyplot as plt

NB_GENERATION = 1000

generations = [i for i in range(0, NB_GENERATION)]
epsilon = 0.5
eps_dec = 0.006
eps_history = []
for i in range(0, NB_GENERATION):
    epsilon = epsilon - (epsilon * eps_dec)
    eps_history.append(epsilon)

fig, ax = plt.subplots()
ax.plot(generations, eps_history, label="epsilon")
plt.show()
