import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from typing import Sequence
from ied import Ied


def configure_relays_trip_figure(figure: Figure) -> None:
    for ax in figure.get_axes():
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize='small', handlelength=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])


def plot_51f_50f_32f_trips(ied: Ied, title: str, resample_limit: float = 0.7) -> None:
    chopped_time = [t for t in ied._mimic_filtered_signals.t if t < resample_limit]
    state_50F = {phase: (ied._mimic_filtered_signals.t >= ied._trips["50F"][phase]) for phase in ["ia", "ib", "ic"]}
    state_51F = {phase: (ied._mimic_filtered_signals.t >= ied._trips["51F"][phase]) for phase in ["ia", "ib", "ic"]}

    figure, axis = plt.subplots(9, 1, figsize=(10, 15))

    for i, phase in enumerate(['ia', 'ib', 'ic']):  # Relés 51F
        axis[i].plot(
            ied._mimic_filtered_signals.t, state_51F[phase],
            label=f'51{chr(65+i)}',
            linewidth=2,
            color=['black', 'blue', 'red'][i]
        )

    for i, phase in enumerate(['ia', 'ib', 'ic']):  # Relés 50F
        axis[i + 3].plot(
            ied._mimic_filtered_signals.t, state_50F[phase],
            label=f'50{chr(65+i)}',
            linewidth=2,
            color=['black', 'blue', 'red'][i]
        )

    for i, phase in enumerate(['a', 'b', 'c']):  # Relés 32F
        axis[i + 6].plot(
            chopped_time, ied._trips["32F"][phase][: len(chopped_time)],
            label=f'32{chr(65+i)}',
            linewidth=2,
            color=['black', 'blue', 'red'][i]
        )

    configure_relays_trip_figure(figure)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)


def plot_51n_50n_32n_67f_67n_trips(ied: Ied, title: str, resample_limit: float = 0.7) -> None:
    chopped_time = [t for t in ied._mimic_filtered_signals.t if t < resample_limit]
    state_50N = ied._mimic_filtered_signals.t >= ied._trips["50N"]["neutral"]
    state_51N = ied._mimic_filtered_signals.t >= ied._trips["51N"]["neutral"]

    figure, axis = plt.subplots(8, 1, figsize=(10, 15))

    axis[0].plot(  # Relé 51N
        ied._mimic_filtered_signals.t, state_51N, label='51N',
        linewidth=2,
        color='black'
    )

    axis[1].plot(  # Relé 50N
        ied._mimic_filtered_signals.t, state_50N,
        label='50N',
        linewidth=2,
        color='black'
    )

    axis[2].plot(  # Relé 32N
        chopped_time, ied._trips["32N"]["neutral"][:len(chopped_time)],
        label='32N',
        linewidth=2,
        color='black'
    )

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i + 3].plot(  # Relés 67F
            ied._mimic_filtered_signals.t, ied._trips["67F"][phase],
            label=f'67{chr(65+i)}',
            linewidth=2,
            color=['black', 'blue', 'red'][i])

    axis[6].plot(  # Relé 67N
        ied._mimic_filtered_signals.t, ied._trips["67N"]["neutral"],
        label='67N',
        linewidth=2,
        color='black'
    )

    axis[7].plot(
        ied._mimic_filtered_signals.t, ied.trip_signal,
        label='Trip signal',
        linewidth=2,
        color='black'
    )

    configure_relays_trip_figure(figure)
    axis[7].set_xticks(np.linspace(0, resample_limit, 8))
    axis[7].set_xlabel('Time (s)')
    axis[7].xaxis.set_visible(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)


# Função para criar o gráfico polar


def plot_32_polar_regions(v_pol_angles: Sequence[float], i_op_angles: Sequence[float], beta: float) -> None:
    figure = plt.figure(figsize=(10, 10))
    grid_spec = GridSpec(3, 2, height_ratios=[1, 1, 0.5])

    ax1 = figure.add_subplot(grid_spec[0, 0], polar=True)
    ax2 = figure.add_subplot(grid_spec[0, 1], polar=True)
    ax3 = figure.add_subplot(grid_spec[1, :], polar=True)
    axes = [ax1, ax2, ax3]
    legend_positions = [(1.1, 1.0), (1.1, 1.0), (0.5, -0.2)]

    for ax, v_pol_angle, i_op_angle in zip(axes, v_pol_angles, i_op_angles):
        ax.set_thetamin(0)  # type: ignore
        ax.set_thetamax(360)  # type: ignore
        ax.set_rlabel_position(0)  # type: ignore
        ax.set_yticklabels([])

        handles, labels = [], []
        start_angle = v_pol_angle - 90 + beta
        end_angle = v_pol_angle + 90 + beta
        theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
        # theta = np.linspace(np.radians(v_pol_angle - 90 + beta), np.radians(v_pol_angle + 90 + beta), 100)
        radius = np.ones_like(theta)
        region = ax.fill(theta, radius, color='lightblue', alpha=0.3, hatch='//')

        handles.append(region[0])
        labels.append(f"{start_angle:.2f}° ≤ $R_{{op}}$ ≤ {end_angle:.2f}°")

        ax.annotate('', xy=(np.radians(v_pol_angle), 1.05), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.annotate('', xy=(np.radians(i_op_angle), 1.05), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        handles.extend([
            Line2D([0], [0], color='black', lw=2, marker='>', markersize=8),
            Line2D([0], [0], color='red', lw=2, marker='>', markersize=8)
        ])
        labels.extend([r"$V_{pol}$ angle", r"$I_{op}$ angle"])

        idx = axes.index(ax)
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=legend_positions[idx],
                  fontsize=8, framealpha=1, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.show(block=True)


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
