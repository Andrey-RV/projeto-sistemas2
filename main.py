import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ied import Ied
from pprint import pprint

# Constantes para o processo de resampling
FUNDAMENTAL_PERIOD = 1 / 60
DESIRED_SAMPLE_RATE = 16

# Impedâncias das linhas de transmissão
R1 = 0.0246 * 250
XL1 = 0.3219 * 250
R0 = 0.376 * 250
XL0 = 1.411 * 250

# Declaração das correntes de ajuste dos relés
PHASE_TIMED_ADJUST_CURRENT = 3
PHASE_INSTA_ADJUST_CURRENT = {'c': 5.03, 'b': 5.49, 'a': 6.05, 'b\'': 11.68, 'c\'': 14.71, 'd\'': 19.87}
NEUTRAL_TIMED_ADJUST_CURRENT = 0.5
NEUTRAL_INSTA_ADJUST_CURRENT = {'c': 1.01, 'b': 1.10, 'a': 1.21, 'b\'': 2.34, 'c\'': 2.94, 'd\'': 3.97}

# Constante para o ângulo de inclinação do relé 21
INCLINATION_ANGLE = np.radians(70)

# Características moh para as zonas de proteção
ZONE1C = [0.1247, 0.4654, 0.9636]
ZONE2C = [0.2200, 0.8212, 1.7004]
ZONE3C = [0.2934, 1.0950, 2.2672]


z1 = R1 + 1j * XL1
z0 = R0 + 1j * XL0


signals = {}
for fault in range(1, 3):
    for bus in range(1, 3):
        signals[f'bus{bus + 1}_fault{fault}'] = pd.read_csv(f"./Registros/Registro{fault}/1Reg{bus}.dat", delimiter='\s+',
                                                            names=['step', 't', 'Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic'])

# Cálculo do período de amostragem e do fator de decimação
sampling_period = signals['bus2_fault1']['t'][1] - signals['bus2_fault1']['t'][0]
md = FUNDAMENTAL_PERIOD / (DESIRED_SAMPLE_RATE * sampling_period)


# Função para criar o objeto IED com os sinais de um determinado barramento e falta (evitar repetição de código)
def create_ied(bus, fault):
    ied = Ied(
        va=signals[f'bus{bus}_fault{fault}']['Va'],
        vb=signals[f'bus{bus}_fault{fault}']['Vb'],
        vc=signals[f'bus{bus}_fault{fault}']['Vc'],
        ia=signals[f'bus{bus}_fault{fault}']['Ia'],
        ib=signals[f'bus{bus}_fault{fault}']['Ib'],
        ic=signals[f'bus{bus}_fault{fault}']['Ic'],
        t=signals[f'bus{bus}_fault{fault}']['t'],
        sampling_period=sampling_period,
        b=1.599e3, c=1.279e6, md=md, R=R1, XL=XL1,
        estimator_sample_rate=16, RTC=1
    )
    return ied


# Função para adicionar os relés 21 ao objeto IED (evitar repetição de código)
def add_function_21(ied: Ied, inclination_angle, zones_impedances, z1, z0):
    ied.add_21(inclination_angle=inclination_angle, zones_impedances=zones_impedances, line_z1=z1, line_z0=z0)
    return


def add_67(ied: Ied, gamma_phase, gamma_neutral):
    ied.add_67(type='phase', gamma=gamma_phase, timed_adjust_current=PHASE_TIMED_ADJUST_CURRENT,
               insta_adjust_current=PHASE_INSTA_ADJUST_CURRENT, curve='IEEE_moderately_inverse')
    ied.add_67(type='neutral', gamma=gamma_neutral, timed_adjust_current=NEUTRAL_TIMED_ADJUST_CURRENT,
               insta_adjust_current=NEUTRAL_INSTA_ADJUST_CURRENT, curve='IEEE_moderately_inverse')


# Função para plotar a primeira parte dos diagramas lógicos (50F e 51F)
def plot_part1(ied: Ied, title, resample_limit=0.7):
    resampled_time = [ied.time[i] for i in range(len(ied.time)) if ied.time[i] < resample_limit]  # Limita o eixo do tempo
    figure, axis = plt.subplots(9, 1, figsize=(10, 15))

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i].plot(ied.time, ied.logical_state_51F[phase], label=f'51{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i + 3].plot(ied.time, ied.logical_state_50F[phase], label=f'50{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for i, phase in enumerate(['a', 'b', 'c']):
        axis[i + 6].plot(resampled_time, ied.trip_permission_32F[phase][:len(resampled_time)], label=f'32{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    # Configurações gerais dos eixos, remoção das bordas, legendas, limites e ticks etc.
    for ax in figure.get_axes():
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize='small', handlelength=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])

    plt.suptitle(title)
    plt.show()


# Função para plotar a segunda parte dos diagramas lógicos (32N, 67F, 67N e trip signal)
def plot_part2(ied: Ied, title, resample_limit=0.7):
    resampled_time = [ied.time[i] for i in range(len(ied.time)) if ied.time[i] < resample_limit]  # Limita o eixo do tempo
    figure, axis = plt.subplots(8, 1, figsize=(10, 15))

    axis[0].plot(ied.time, ied.logical_state_51N, label='51N', linewidth=2, color='black')
    axis[1].plot(ied.time, ied.logical_state_50N, label='50N', linewidth=2, color='black')
    axis[2].plot(resampled_time, ied.trip_permission_32N[:len(resampled_time)], label='32N', linewidth=2, color='black')

    for i, phase in enumerate(['a', 'b', 'c']):
        axis[i + 3].plot(ied.time, ied.trip_permission_67F[phase], label=f'67{chr(65+i)}', linewidth=2,
                         color=['black', 'blue', 'red'][i])

    axis[6].plot(ied.time, ied.trip_permission_67N, label='67N', linewidth=2, color='black')
    axis[7].plot(ied.time, ied.trip_signal, label='Trip signal', linewidth=2, color='black')

    # Configurações gerais dos eixos, remoção das bordas, legendas, limites e ticks etc.
    for ax in figure.get_axes():
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.legend(loc='center left', fontsize='xx-small', handlelength=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])

    axis[7].set_xticks(np.linspace(0, resample_limit, 8))
    axis[7].set_xlabel('Time (s)')
    axis[7].xaxis.set_visible(True)

    plt.suptitle(title)
    plt.show()


# Criação dos IEDs para cada barramento e falta, adição dos relés 67 e 21
for bus in range(2, 4):
    gamma_phase = 0.069 if bus == 2 else 0.173  # multiplicadores de tempo para as unidades 51F
    gamma_neutral = 0.240 if bus == 2 else 0.312  # multiplicadores de tempo para as unidades 51N

    for fault in range(1, 3):
        ied = create_ied(bus, fault)
        add_67(ied, gamma_phase, gamma_neutral)
        # plot_part1(ied, f'Bus {bus} Fault {fault}')
        # plot_part2(ied, f'Bus {bus} Fault {fault}')
        add_function_21(ied, INCLINATION_ANGLE, [0.85 * z1, 1.5 * z1, 2 * z1], z1, z0)

        # Plot das zonas de proteção e trajetórias das impedâncias
        for unit in ['at', 'bt', 'ct', 'ab', 'bc', 'ca']:
            dx = np.diff(np.real(ied.measured_impedances[unit]))
            dy = np.diff(np.imag(ied.measured_impedances[unit]))
        fig, ax = plt.subplots()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal', adjustable='datalim')
        ax.scatter(np.real(ied.measured_impedances[unit]), np.imag(ied.measured_impedances[unit]), color='black', s=15)
        ax.plot(np.real(ied.measured_impedances[unit]), np.imag(ied.measured_impedances[unit]), color='black', linestyle='--')
        ax.quiver(np.real(ied.measured_impedances[unit])[:-1], np.imag(ied.measured_impedances[unit])[:-1], dx, dy,
                  angles='xy', scale_units='xy', scale=1, color='black', width=0.004)
        ax.title.set_text(f'Barra {bus}, Falta {fault}, Unidade {unit.upper()}')

        zone1_circle = plt.Circle((ZONE1C[0], ZONE1C[1]), ZONE1C[2], color='blue', fill=False, label='Zona 1')
        ax.add_artist(zone1_circle)
        zone2_circle = plt.Circle((ZONE2C[0], ZONE2C[1]), ZONE2C[2], color='green', fill=False, label='Zona 2')
        ax.add_artist(zone2_circle)
        zone3_circle = plt.Circle((ZONE3C[0], ZONE3C[1]), ZONE3C[2], color='red', fill=False, label='Zona 3')
        ax.add_artist(zone3_circle)
        ax.title.set_text(f'Unidade {unit.upper()} na Barra {bus}')

        plt.legend()
        plt.show()

        # Cálculo das impedâncias e distâncias observadas pelas unidades 21
        real_impedance = ied.measured_impedances[unit][-1] * ((500e3/115) / (300 / 5))
        with open("output.txt", "a") as file:
            file.write(f'*Falta {"ACT" if fault == 1 else "BT"} na barra {bus}*\n')
            file.write(f'Impedância vista pela unidade {unit.upper()} na barra {bus}: {ied.measured_impedances[unit][-1]} ohms\n')
            file.write(f'Impedância calculada ao ponto da falta pela unidade {unit.upper()} na barra {bus}: {real_impedance} ohms\n')
            file.write(f'Distância observada pela unidade {unit.upper()} na barra {bus}: {np.abs(real_impedance) / np.abs(0.0246 + 1j * 0.3219)} km\n')
            file.write(f'Resultado válido? {"*Sim*" if np.abs(real_impedance) <= np.abs(2*z1) else "*Não*"}\n')
            file.write('\n')

        # Salvar os resultados do comparador cosseno
        with open(f"./Registros/Registro{fault}/bus{bus}_fault{fault}_21.txt", 'w') as f:
            print(f"Bus {bus}, fault {fault}:", file=f)
            pprint(ied.distance_trip_signals, stream=f)
