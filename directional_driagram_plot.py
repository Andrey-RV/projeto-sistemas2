import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def create_polar_plot(ax, v_pol, i_op):
    line_v_pol, = ax.plot([0, np.radians(v_pol)], [0, 1], color='black', linewidth=2, label=r'$\dot{V}_{pol}$' + f' ({v_pol}°)')
    line_i_op, = ax.plot([0, np.radians(i_op)], [0, 1], color='red', linewidth=2, label=r'$\dot{I}_{op}$' + f' ({i_op}°)')

    theta = np.linspace(np.radians(v_pol - 60), np.radians(v_pol + 120), 100)
    radius = np.ones_like(theta)
    hatch_fill = ax.fill(theta, radius, color='lightblue', alpha=0.5, hatch='//')

    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_rlabel_position(0)
    ax.set_yticklabels([])

    ax.annotate('', xy=(np.radians(v_pol), 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(np.radians(i_op), 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    legend_text = f"{v_pol - 60:.2f}° ≤ " + r"$R_{op}$" + f" ≤ {v_pol + 120:.2f}°"
    handles, labels = ax.get_legend_handles_labels()
    handles.append(hatch_fill[0])
    labels.append(legend_text)

    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.4, 1.7), fontsize=8, framealpha=1, facecolor='white', edgecolor='black', title_fontsize='x-small')


fig = plt.figure(figsize=(10, 7))
gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5])

ax1 = fig.add_subplot(gs[0, 0], polar=True)
create_polar_plot(ax1, -85.58, -173.85)

ax2 = fig.add_subplot(gs[0, 1], polar=True)
create_polar_plot(ax2, 154.42, 66.15)

ax3 = fig.add_subplot(gs[1, :], polar=True)
create_polar_plot(ax3, 34.42, -53.85)

plt.show()
