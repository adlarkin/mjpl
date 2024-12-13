import matplotlib.pyplot as plt
import pickle

labels = ['IK', 'RRT', 'RRT + Shortcutting']

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

fig, axs = plt.subplots(1, 2)

axs[0].boxplot(
    [data['IK_valid_solution_times'], data['RRT_valid_solution_times'], data['RRT_shortcutting_valid_solution_times']],
    tick_labels=labels,
)
axs[0].set_title('Solution Time (s)')

axs[1].boxplot(
    [data['IK_valid_solution_lengths'], data['RRT_valid_solution_lengths'], data['RRT_shortcutting_valid_solution_lengths']],
    tick_labels=labels,
)
axs[1].set_title('EE Path Length (meters)')

fig_bar, ax_bar = plt.subplots()
ax_bar.bar(
    x=[0, 1],
    height=[data['IK_success_rate'], data['RRT_success_rate']],
    tick_label=labels[:2],
    color=['tab:red', 'tab:blue'],
)
ax_bar.set_title('Success Rate (100 Attempts)')

plt.show()
