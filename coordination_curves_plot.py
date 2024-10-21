import matplotlib.pyplot as plt
import numpy as np


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
