import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ied import Ied

# Constantes para o processo de resampling
FUNDAMENTAL_PERIOD = 1 / 60
DESIRED_SAMPLE_RATE = 16

# Impedâncias das linhas de transmissão
R1 = 0.0246 * 250
XL1 = 0.3219 * 250
R0 = 0.376 * 250
XL0 = 1.411 * 250

# Constante para o ângulo de inclinação do relé 21
INCLINATION_ANGLE = np.radians(70)


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


# Criação dos IEDs para cada barramento e falta, adição dos relés 21
zone1c = [0.1247, 0.4654, 0.9636]
zone2c = [0.2200, 0.8212, 1.7004]
zone3c = [0.2934, 1.0950, 2.2672]

for bus in range(2, 4):
    for fault in range(1, 3):
        ied = create_ied(bus, fault)
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

        zone1_circle = plt.Circle((zone1c[0], zone1c[1]), zone1c[2], color='blue', fill=False, label='Zona 1')
        ax.add_artist(zone1_circle)
        zone2_circle = plt.Circle((zone2c[0], zone2c[1]), zone2c[2], color='green', fill=False, label='Zona 2')
        ax.add_artist(zone2_circle)
        zone3_circle = plt.Circle((zone3c[0], zone3c[1]), zone3c[2], color='red', fill=False, label='Zona 3')
        ax.add_artist(zone3_circle)
        ax.title.set_text(f'Unidade {unit.upper()} na Barra {bus}')

        plt.legend()
        plt.show()

        # Cálculo das impedâncias e distâncias observadas pelas unidades 21
        #     real_impedance = ied.measured_impedances[unit][-1] * ((500e3/115) / (300 / 5))
        #     with open("output.txt", "a") as file:
        #         file.write(f'*Falta {"ACT" if fault == 1 else "BT"} na barra {bus}*\n')
        #         file.write(f'Impedância vista pela unidade {unit.upper()} na barra {bus}: {ied.measured_impedances[unit][-1]} ohms\n')
        #         file.write(f'Impedância calculada ao ponto da falta pela unidade {unit.upper()} na barra {bus}: {real_impedance} ohms\n')
        #         file.write(f'Distância observada pela unidade {unit.upper()} na barra {bus}: {np.abs(real_impedance) / np.abs(0.0246 + 1j * 0.3219)} km\n')
        #         file.write(f'Resultado válido? {"*Sim*" if np.abs(real_impedance) <= np.abs(2*z1) else "*Não*"}\n')
        #         file.write('\n')

        # Salvar os resultados do comparador cosseno
        # with open(f"./Registros/Registro{fault}/bus{bus}_fault{fault}_21.txt", 'w') as f:
        #     print(f"Bus {bus}, fault {fault}:", file=f)
        #     pprint(ied.distance_trip_signals, stream=f)
