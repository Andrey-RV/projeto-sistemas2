from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from typing import Sequence, List, Dict, Tuple
from ied import Ied
from relays import Curve


def configure_relays_trip_figure(figure: Figure) -> None:
    """
    Remove as bordas e os ticks dos eixos, e configura a legenda e o eixo y para todos os subgráficos da figura.
    """
    for ax in figure.get_axes():
        # Oculta todas as bordas (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize='small', handlelength=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])


def plot_51f_50f_32f_trips(ied: Ied, title: str, resample_limit: float = 0.7) -> None:
    """
    Plota as curvas de atuação dos relés 51F, 50F e 32F.

    Parâmetros:
        ied: Instância de Ied contendo os dados dos sinais e dos disparos.
        title: Título do gráfico.
        resample_limit: Limite de tempo para amostragem dos dados (padrão: 0.7).
    """
    t_array = np.array(ied._mimic_filtered_signals.t)
    chopped_time = t_array[t_array < resample_limit]

    state_50F = {phase: (t_array >= ied._trips["50F"][phase]) for phase in ["ia", "ib", "ic"]}
    state_51F = {phase: (t_array >= ied._trips["51F"][phase]) for phase in ["ia", "ib", "ic"]}

    fig, axes = plt.subplots(9, 1, figsize=(10, 15))
    colors = ['black', 'blue', 'red']

    # Plota relés 51F
    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axes[i].plot(t_array, state_51F[phase],
                     label=f'51{chr(65+i)}',
                     linewidth=2,
                     color=colors[i])

    # Plota relés 50F
    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axes[i + 3].plot(t_array, state_50F[phase],
                         label=f'50{chr(65+i)}',
                         linewidth=2,
                         color=colors[i])

    # Plota relés 32F
    for i, phase in enumerate(['a', 'b', 'c']):
        axes[i + 6].plot(chopped_time, ied._trips["32F"][phase][:len(chopped_time)],
                         label=f'32{chr(65+i)}',
                         linewidth=2,
                         color=colors[i])

    configure_relays_trip_figure(fig)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)


def plot_51n_50n_32n_67f_67n_trips(ied: Ied, title: str, resample_limit: float = 0.7) -> None:
    """
    Plota as curvas de atuação dos relés 51N, 50N, 32N, 67F e 67N.

    Parâmetros:
        ied: Instância de Ied contendo os dados dos sinais e dos disparos.
        title: Título do gráfico.
        resample_limit: Limite de tempo para amostragem dos dados (padrão: 0.7).
    """
    t_array = np.array(ied._mimic_filtered_signals.t)
    chopped_time = t_array[t_array < resample_limit]

    state_50N = t_array >= ied._trips["50N"]["neutral"]
    state_51N = t_array >= ied._trips["51N"]["neutral"]

    fig, axes = plt.subplots(8, 1, figsize=(10, 15))

    # Plota relé 51N
    axes[0].plot(t_array, state_51N,
                 label='51N',
                 linewidth=2,
                 color='black')

    # Plota relé 50N
    axes[1].plot(t_array, state_50N,
                 label='50N',
                 linewidth=2,
                 color='black')

    # Plota relé 32N
    axes[2].plot(chopped_time, ied._trips["32N"]["neutral"][:len(chopped_time)],
                 label='32N',
                 linewidth=2,
                 color='black')

    colors = ['black', 'blue', 'red']
    # Plota relés 67F para as fases ia, ib e ic
    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axes[i + 3].plot(t_array, ied._trips["67F"][phase],
                         label=f'67{chr(65+i)}',
                         linewidth=2,
                         color=colors[i])

    # Plota relé 67N
    axes[6].plot(t_array, ied._trips["67N"]["neutral"],
                 label='67N',
                 linewidth=2,
                 color='black')

    # Plota sinal de disparo (Trip signal)
    axes[7].plot(t_array, ied.trip_signal,
                 label='Trip signal',
                 linewidth=2,
                 color='black')

    configure_relays_trip_figure(fig)
    axes[7].set_xticks(np.linspace(0, resample_limit, 8))
    axes[7].set_xlabel('Time (s)')
    axes[7].xaxis.set_visible(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)


def plot_32_polar_regions(v_pol_angles: Sequence[float], i_op_angles: Sequence[float], beta: float) -> None:
    """
    Plota as regiões polares para os relés 32 utilizando os ângulos polares de tensão e os ângulos de corrente
    operativa.

    Parâmetros:
        v_pol_angles: Sequência de ângulos polares de tensão.
        i_op_angles: Sequência de ângulos de corrente operativa.
        beta: Valor de ajuste angular.
    """
    fig = plt.figure(figsize=(10, 10))
    grid_spec = GridSpec(3, 2, height_ratios=[1, 1, 0.5])

    ax1 = fig.add_subplot(grid_spec[0, 0], polar=True)
    ax2 = fig.add_subplot(grid_spec[0, 1], polar=True)
    ax3 = fig.add_subplot(grid_spec[1, :], polar=True)
    axes = [ax1, ax2, ax3]
    legend_positions = [(1.1, 1.0), (1.1, 1.0), (0.5, -0.2)]

    for idx, (ax, v_pol_angle, i_op_angle) in enumerate(zip(axes, v_pol_angles, i_op_angles)):
        ax.set_thetamin(0)  # type: ignore
        ax.set_thetamax(360)  # type: ignore
        ax.set_rlabel_position(0)  # type: ignore
        ax.set_yticklabels([])  # type: ignore

        start_angle = v_pol_angle - 90 + beta
        end_angle = v_pol_angle + 90 + beta
        theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
        radius = np.ones_like(theta)
        region = ax.fill(theta, radius, color='lightblue', alpha=0.3, hatch='//')

        handles = [region[0]]
        labels = [f"{start_angle:.2f}° ≤ $R_{{op}}$ ≤ {end_angle:.2f}°"]

        ax.annotate('', xy=(np.radians(v_pol_angle), 1.05), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.annotate('', xy=(np.radians(i_op_angle), 1.05), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        handles.extend([
            Line2D([0], [0], color='black', lw=2, marker='>', markersize=8),  # type: ignore
            Line2D([0], [0], color='red', lw=2, marker='>', markersize=8)  # type: ignore
        ])
        labels.extend([r"$V_{pol}$ angle", r"$I_{op}$ angle"])

        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=legend_positions[idx],
                  fontsize=8, framealpha=1, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.show(block=True)


def _configure_chart() -> None:
    """
    Configura o gráfico com rótulos, escalas logarítmicas, legenda e grade.
    """
    plt.xlabel('Corrente de primário (A)')
    plt.ylabel('Tempo de atuação (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()


def get_trip_curve(
    timed_adjust_current: float,
    insta_adjust_current: float,
    gamma: float,
    curve: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula a curva de atuação do relé com base nos parâmetros fornecidos.

    Parâmetros:
        timed_adjust_current: Corrente de ajuste temporizado.
        insta_adjust_current: Corrente de ajuste instantâneo.
        gamma: Valor de gamma para o cálculo.
        curve: Identificador da curva (ex.: 'IEEE_moderately_inverse').

    Retorna:
        Uma tupla contendo os arrays de correntes e os tempos de atuação.
    """
    k, a, c = Curve[curve.upper()].value
    currents = np.arange(0.01 * timed_adjust_current, 25, 0.01)
    curve_common_term = gamma * (k / ((currents / timed_adjust_current) ** a - 1) + c)
    timed_times = np.where(currents <= timed_adjust_current, np.inf, curve_common_term)
    insta_times = np.where(currents >= insta_adjust_current, 1e-3, timed_times)
    return currents, insta_times


def plot_trip_curves(relays: List[str],
                     timed_adjust_current: float,
                     insta_adjust_current: Dict[str, float],
                     gamma: Dict[str, float],
                     title: str,
                     curve: str = 'IEEE_moderately_inverse',
                     multiplier: float = 60) -> None:
    """
    Plota as curvas de atuação dos relés para uma lista de relés.

    Parâmetros:
        relays: Lista de identificadores dos relés.
        timed_adjust_current: Corrente para ajuste temporizado.
        insta_adjust_current: Dicionário mapeando cada relé para sua corrente de ajuste instantâneo.
        gamma: Dicionário mapeando cada relé para seu valor de gamma.
        title: Título do gráfico.
        curve: Identificador da curva de atuação (padrão: 'IEEE_moderately_inverse').
        multiplier: Fator de multiplicação para os valores de corrente (padrão: 60).
    """
    plt.figure(figsize=(10, 10))
    for relay in relays:
        currents, trip_times = get_trip_curve(
            timed_adjust_current,
            insta_adjust_current[relay],
            gamma[relay],
            curve=curve
        )
        plt.plot(currents * multiplier, trip_times, label=f'Relé {relay.upper()}')
    _configure_chart()
    plt.title(title)
    plt.show(block=True)
