import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anti_aliasing import AntiAliasingFilter
from mimic_filter import MimicFilter
from fourier_filter import PhasorEstimator


R = 0.0246 * 200  # 200 km de linha de transmissão com resistência de 0.0246 ohm/km.
XL = 0.3219 * 200  # 200 km de linha de transmissão com reatância de 0.3219 ohm/km.
L = XL / (2 * np.pi * 60)  # Indutância da linha @ 60 Hz.


def resample(signal, md: int):
    '''
    Reamostra um sinal de acordo com o fator de decimação md fatiando o sinal original.
    Args:
        signal: Sinal a ser reamostrado.
        md (int): Fator de decimação.
    Returns:
        Sinal reamostrado.
    '''
    return signal[::md].reshape(-1,)  # Reshape necessário pois o fatiamento retorna um array 2D.


# Leitura dos sinais do emissor. Colunas renomeadas pois o arquivo não possui cabeçalho.
emitter_signals = pd.read_csv("./Atividade_01/1Reg1.dat", delimiter='\s+',
                              names=['1', 't', '3', '4', '5', '6', 'va', 'vb', 'vc',
                                     '10', 'ia', 'ib', 'ic', '14', '15', '16', '17', '18'])

original_sampling_period = emitter_signals['t'][1] - emitter_signals['t'][0]
va = emitter_signals['va']
ia = emitter_signals['ia']

# Filtragem anti-aliasing.
aa_filter_va = AntiAliasingFilter(period=original_sampling_period, signal=va, b=1.599e3, c=1.279e6)
aa_filter_ia = AntiAliasingFilter(period=original_sampling_period, signal=ia, b=1.599e3, c=1.279e6)
aa_filter_va.apply_filter()
aa_filter_ia.apply_filter()
filtered_va = aa_filter_va.filtered_signal
filtered_ia = aa_filter_ia.filtered_signal

md = int(1e-3 / original_sampling_period)  # md = Delta_t2 / Delta_t1. Como não são múltiplos e t=16.67, escolhemos Delta_t1=1ms.
resampled_va = resample(filtered_va, md)
resampled_ia = resample(filtered_ia, md)
new_time_points = list(emitter_signals['t'][::md])  # Conversão do np.array para lista para ser utilizado na plotagem.
new_sampling_period = new_time_points[-1] - new_time_points[-2]

# Figura representando va, ia, os sinais filtrados (anti_aliasing) e filtrados e reamostrados.
figure, axis = plt.subplots(2, 1)
axis[0].plot(emitter_signals['t'], va, label=r'$v_a(n\dot\Delta t_1)$', color='blue')
axis[0].plot(emitter_signals['t'], filtered_va, label=r'$v_a(n\dot\Delta t_1)$ filtrado', color='red')
axis[0].scatter(new_time_points, resampled_va, label=r'$v_a(k\dot M_d\Delta t_1)$ (filtrado e reamostrado)', color='black', s=8)
axis[0].set_title('Tensão na fase A do Emissor')
axis[0].set_xlabel('Tempo (s)')
axis[0].set_ylabel('Tensão (V)')
axis[0].grid()
axis[0].legend(prop={'size': 7})
axis[1].plot(emitter_signals['t'], ia, label=r'$i_a(n\dot\Delta t_1)$', color='blue')
axis[1].plot(emitter_signals['t'], filtered_ia, label=r'$i_a(n\dot\Delta t_1)$ filtrado', color='red')
axis[1].scatter(new_time_points, resampled_ia, label=r'$i_a(k\dot M_d\Delta t_1)$ (filtrado e reamostrado)', color='black', s=8)
axis[1].set_title('Corrente na fase A do Emissor')
axis[1].set_xlabel('Tempo (s)')
axis[1].set_ylabel('Corrente (A)')
axis[1].grid()
axis[1].legend(prop={'size': 7})
plt.show()

# Filtro mímico.
tau = (L / R) / new_sampling_period
mimic_filter_va = MimicFilter(resampled_va, tau, new_sampling_period)
mimic_filter_ia = MimicFilter(resampled_ia, tau, new_sampling_period)
mimic_filter_va.apply_filter()
mimic_filter_ia.apply_filter()
mimic_filtered_va = mimic_filter_va.filtered_signal
mimic_filtered_ia = mimic_filter_ia.filtered_signal

# Figura representando va e ia reamostrados e o sinal filtrado pelo filtro mímico.
figure, axis = plt.subplots(2, 1)
axis[0].plot(new_time_points, resampled_va, label=r'$v_a(kM_d\dot\Delta t_1)$', color='blue',
             marker='o', markersize=3, markerfacecolor='black')
mimic_filter_va.apply_filter()
axis[0].plot(new_time_points, mimic_filtered_va, label=r'$v_a(kM_d\dot\Delta t_1)$' + ' pós filtro mímico', color='red',
             marker='o', markersize=3, markerfacecolor='red')
axis[0].set_title('Tensão na fase A do Emissor')
axis[0].set_xlabel('Tempo (s)')
axis[0].set_ylabel('Tensão (V)')
axis[0].grid()
axis[0].legend()
axis[1].plot(new_time_points, resampled_ia, label=r'$i_a(kM_d\dot\Delta t_1)$', color='blue',
             marker='o', markersize=3, markerfacecolor='black')
mimic_filter_ia.apply_filter()
axis[1].plot(new_time_points, mimic_filtered_ia, label=r'$i_a(kM_d\dot\Delta t_1)$' + ' pós filtro mímico', color='red',
             marker='o', markersize=3, markerfacecolor='red')
axis[1].set_title('Corrente na fase A do Emissor')
axis[1].set_xlabel('Tempo (s)')
axis[1].set_ylabel('Corrente (A)')
axis[1].grid()
axis[1].legend()
plt.show()

# Estimação de fasores.
phasor_va = PhasorEstimator(mimic_filtered_va, samples_per_cycle=16)
phasor_ia = PhasorEstimator(mimic_filtered_ia, samples_per_cycle=16)
phasor_va.estimate()
phasor_ia.estimate()

# Figura representando o sinal após filtro mímico e o fasor estimado para va.
figure, axis = plt.subplots(2, 1)
axis[0].plot(new_time_points, mimic_filtered_va, label=r'$v_a(kM_d\dot\Delta t_1)$' + 'pós filtro mímico', color='blue',
             marker='o', markersize=3, markerfacecolor='blue')
axis[0].plot(new_time_points, phasor_va.amplitude[:2301], label=r'$|V_a|$', color='red')
axis[0].set_title('Tensão na fase A do Emissor')
axis[0].set_xlabel('Tempo (s)')
axis[0].set_ylabel('Tensão (V)')
axis[0].grid()
axis[0].legend()
axis[1].plot(new_time_points, phasor_va.phase[:2301], label=r'$\phi V_a$', color='red')
axis[1].set_xlabel('Tempo (s)')
axis[1].set_ylabel('Fase (graus)')
axis[1].legend()
axis[1].grid()
plt.show()

# Figura representando o sinal após filtro mímico e o fasor estimado para ia.
figure, axis = plt.subplots(2, 1)
axis[0].plot(new_time_points, mimic_filtered_ia, label=r'$i_a(kM_d\dot\Delta t_1)$' + 'pós filtro mímico', color='blue',
             marker='o', markersize=3, markerfacecolor='blue')
axis[0].plot(new_time_points, phasor_ia.amplitude[:2301], label=r'$|I_a|$', color='red')
axis[0].set_title('Corrente na fase A do Emissor')
axis[0].set_xlabel('Tempo (s)')
axis[0].set_ylabel('Corrente (A)')
axis[0].grid()
axis[0].legend()
axis[1].plot(new_time_points, phasor_ia.phase[:2301], label=r'$\phi I_a$', color='red')
axis[1].set_xlabel('Tempo (s)')
axis[1].set_ylabel('Fase (graus)')
axis[1].legend()
axis[1].grid()
plt.show()

# # Densidade espectral de potência e média dos sinais.
# figure, axis = plt.subplots(2, 1)
# axis[0].psd(resampled_va.flatten(), NFFT=2048, Fs=1/new_sampling_period, label=r'$v_a(kM_d\dot\Delta t_1)$')
# axis[0].psd(mimic_filtered_va.flatten(), NFFT=2048, Fs=1/new_sampling_period, label=r'$v_a(kM_d\dot\Delta t_1)$ pós filtro mímico')
# axis[0].set_title('Densidade Espectral de Potência da Tensão na Fase A do Emissor')
# axis[0].set_xlabel('Frequência (Hz)')
# axis[0].set_ylabel('Potência (dB)')
# axis[0].grid()
# axis[0].legend()
# axis[1].psd(resampled_ia.flatten(), NFFT=2048, Fs=1/new_sampling_period, label=r'$i_a(kM_d\dot\Delta t_1)$')
# axis[1].psd(mimic_filtered_ia.flatten(), NFFT=2048, Fs=1/new_sampling_period, label=r'$i_a(kM_d\dot\Delta t_1)$ pós filtro mímico')
# axis[1].set_title('Densidade Espectral de Potência da Corrente na Fase A do Emissor')
# axis[1].set_xlabel('Frequência (Hz)')
# axis[1].set_ylabel('Potência (dB)')
# axis[1].grid()
# axis[1].legend()
# plt.show()

# print(f'''
#       Va pré filtro mímico: {np.mean(resampled_va)}
#       Va pós filtro mímico: {np.mean(mimic_filtered_va)}
#       Ia pré filtro mímico: {np.mean(resampled_ia)}
#       Ia pós filtro mímico: {np.mean(mimic_filtered_ia)}
#       ''')
