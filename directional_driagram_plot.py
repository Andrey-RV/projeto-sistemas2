import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# Função para criar o gráfico polar
def create_polar_plot(ax, angle_v_pol, angle_i_op):
    line_v_pol, = ax.plot([0, np.radians(angle_v_pol)], [0, 1], color='black', linewidth=2,
                          label=r'$\dot{V}_{pol}$' + f' ({angle_v_pol}°)')
    line_i_op, = ax.plot([0, np.radians(angle_i_op)], [0, 1], color='red', linewidth=2,
                         label=r'$\dot{I}_{op}$' + f' ({angle_i_op}°)')

    # Criação da área hachurada no gráfico polar para a região de operação beta = 30°
    theta = np.linspace(np.radians(angle_v_pol - 60), np.radians(angle_v_pol + 120), 100)
    radius = np.ones_like(theta)
    hatch_fill = ax.fill(theta, radius, color='lightblue', alpha=0.5, hatch='//')

    # Configurações do gráfico polar
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_rlabel_position(0)
    ax.set_yticklabels([])

    # Adiciona as setas no gráfico polar
    ax.annotate('', xy=(np.radians(angle_v_pol), 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(np.radians(angle_i_op), 1.05), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Adiciona a legenda no gráfico polar
    legend_text = f"{angle_v_pol - 60:.2f}° ≤ " + r"$R_{op}$" + f" ≤ {angle_v_pol + 120:.2f}°"
    handles, labels = ax.get_legend_handles_labels()
    handles.append(hatch_fill[0])
    labels.append(legend_text)

    # Configuração da legenda
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.4, 1.7), fontsize=8, framealpha=1, facecolor='white', edgecolor='black', title_fontsize='x-small')


# Criação da figura e dos subplots
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5])

# Primeiro gráfico polar
ax1 = fig.add_subplot(gs[0, 0], polar=True)
create_polar_plot(ax1, -85.58, -173.85)

# Segundo gráfico polar
ax2 = fig.add_subplot(gs[0, 1], polar=True)
create_polar_plot(ax2, 154.42, 66.15)

# Terceiro gráfico polar
ax3 = fig.add_subplot(gs[1, :], polar=True)
create_polar_plot(ax3, 34.42, -53.85)

plt.show()
