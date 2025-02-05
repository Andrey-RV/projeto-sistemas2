import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def configure_figure(figure):  # type: ignore
    for ax in figure.get_axes():
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize='small', handlelength=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])


def configure_x_axis(axis, resample_limit):
    axis.set_xticks(np.linspace(0, resample_limit, 8))
    axis.set_xlabel('Time (s)')
    axis.xaxis.set_visible(True)


def plot_trips(ied, title, resample_limit=0.7):
    chopped_time = [t for t in ied._mimic_filtered_signals.t if t < resample_limit]
    state_50F = {phase: (ied._mimic_filtered_signals.t >= ied._trips["50F"][phase]) for phase in ["ia", "ib", "ic"]}
    state_51F = {phase: (ied._mimic_filtered_signals.t >= ied._trips["51F"][phase]) for phase in ["ia", "ib", "ic"]}
    state_50N = ied._mimic_filtered_signals.t >= ied._trips["50N"]["neutral"]
    state_51N = ied._mimic_filtered_signals.t >= ied._trips["51N"]["neutral"]

    # Primeira figura com os relés 51F, 50F e 32F
    figure, axis = plt.subplots(9, 1, figsize=(10, 15))

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i].plot(ied._mimic_filtered_signals.t, state_51F[phase],
                     label=f'51{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i + 3].plot(ied._mimic_filtered_signals.t, state_50F[phase],
                         label=f'50{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for i, phase in enumerate(['a', 'b', 'c']):
        axis[i + 6].plot(chopped_time, ied._trips["32F"][phase][: len(chopped_time)],
                         label=f'32{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    configure_figure(figure)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)

    # Segunda figura com os relés 32N, 51N, 50N, 67F e 67N
    figure, axis = plt.subplots(8, 1, figsize=(10, 15))

    axis[0].plot(ied._mimic_filtered_signals.t, state_51N, label='51N', linewidth=2, color='black')
    axis[1].plot(ied._mimic_filtered_signals.t, state_50N, label='50N', linewidth=2, color='black')
    axis[2].plot(chopped_time, ied._trips["32N"]["neutral"]
                 [:len(chopped_time)], label='32N', linewidth=2, color='black')

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i + 3].plot(ied._mimic_filtered_signals.t, ied._trips["67F"][phase],
                         label=f'67{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    axis[6].plot(ied._mimic_filtered_signals.t, ied._trips["67N"]["neutral"], label='67N', linewidth=2, color='black')
    axis[7].plot(ied._mimic_filtered_signals.t, ied.trip_signal, label='Trip signal', linewidth=2, color='black')

    configure_figure(figure)
    configure_x_axis(axis[7], resample_limit)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)


# Função para criar o gráfico polar
def create_polar_plot(ax, angle_v_pol, angle_i_op):

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
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.4, 1.7), fontsize=8,
              framealpha=1, facecolor='white', edgecolor='black', title_fontsize='x-small')


# Função para criar o gráfico de curvas de coordenação do relé 51
def overcurrent_timed(currents, trip_time, is_neutral=False):
    scale = 30 if is_neutral else 180  # divisão da RTC com a corrente de ajuste
    curve_common_term = 0.0515 / ((currents / scale) ** 0.02 - 1) + 0.114
    gamma_values = [('c', 0.001), ('b', 0.240 if is_neutral else 0.069), ('a', 0.494 if is_neutral else 0.148),
                    ('b\'', 0.001), ('c\'', 0.312 if is_neutral else 0.173), ('d\'', 0.658 if is_neutral else 0.375)]

    for relay, gamma in gamma_values:
        trip_time[relay] = np.where(currents <= (0.5 if is_neutral else 3), np.inf, gamma * curve_common_term)


# Função para criar o gráfico de curvas de coordenação do relé 50
def overcurrent_instantaneous(currents, trip_time, is_neutral=False):
    thresholds = [1.01, 1.10, 1.21, 2.34, 2.94, 3.97] if is_neutral else [5.03, 5.49, 6.05, 11.68, 14.71, 19.87]
    currents = currents / 60

    for i, relay in enumerate(['c', 'b', 'a', 'b\'', 'c\'', 'd\'']):
        trip_time[relay] = np.where(currents >= thresholds[i], 1e-4, trip_time[relay])


# Função para criar o gráfico de curvas de coordenação
def plot_trip_times(currents, trip_times, title):
    for relay, label in zip(trip_times, ['Relé C', 'Relé B', 'Relé A', 'Relé B\'', 'Relé C\'', 'Relé D\'']):
        plt.plot(currents, trip_times[relay], label=label)
    plt.xlabel('Corrente de primário (A)')
    plt.ylabel('Tempo de atuação (s)')
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':

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

    # Inicialização dos dicionários vazios de tempos de atuação
    phase_trip_times = {key: np.array([]) for key in ['c', 'b', 'a', 'b\'', 'c\'', 'd\'']}
    neutral_trip_times = {key: np.array([]) for key in ['c', 'b', 'a', 'b\'', 'c\'', 'd\'']}

    # Amostras de corrente e tempo
    fault_currents = np.arange(0, 20 * 60, 0.5)
    times = np.arange(0, 0.1, 0.000001)

    # Cálculo dos tempos de atuação
    overcurrent_timed(fault_currents, phase_trip_times)
    overcurrent_instantaneous(fault_currents, phase_trip_times)
    overcurrent_timed(fault_currents, neutral_trip_times, is_neutral=True)
    overcurrent_instantaneous(fault_currents, neutral_trip_times, is_neutral=True)

    # Plotagem dos tempos de atuação
    plot_trip_times(fault_currents, phase_trip_times, 'Phase Relays')
    plot_trip_times(fault_currents, neutral_trip_times, 'Neutral Relays')
