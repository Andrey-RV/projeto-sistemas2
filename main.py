import pandas as pd
import matplotlib.pyplot as plt
from anti_aliasing import AntiAliasingFilter
from fourier_filter import PhasorEstimator


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

# Estimação de fasores.
phasor_va = PhasorEstimator(resampled_va, samples_per_cycle=16)
phasor_ia = PhasorEstimator(resampled_ia, samples_per_cycle=16)
phasor_va.estimate()
phasor_ia.estimate()

# Figura representando o sinal reamostrado e o fasor estimado para va.
figure, axis = plt.subplots(2, 1)
axis[0].plot(new_time_points, resampled_va, label=r'$v_a(kM_d\dot\Delta t_1)$', color='blue',
             marker='o', markersize=3, markerfacecolor='black')
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

# Figura representando o sinal reamostrado e o fasor estimado para ia.
figure, axis = plt.subplots(2, 1)
axis[0].plot(new_time_points, resampled_ia, label=r'$i_a(kM_d\dot\Delta t_1)$', color='blue',
             marker='o', markersize=3, markerfacecolor='black')
axis[0].plot(new_time_points, phasor_ia.amplitude[:2301], label='amplitude', color='red')
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
