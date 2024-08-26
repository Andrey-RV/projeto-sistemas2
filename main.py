import pandas as pd
import matplotlib.pyplot as plt
from anti_aliasing import AntiAliasingFilter
from fourier_filter import PhasorEstimator


def resample(signal, md):
    return signal[::md].reshape(-1,)


emitter_signals = pd.read_csv("./Atividade_01/1Reg1.dat", delimiter='\s+',
                             names=['1', 't', '3', '4', '5', '6', 'va', 'vb', 'vc', '10', 'ia', 'ib', 'ic', '14', '15', '16', '17', '18'])

original_period = emitter_signals['t'][1] - emitter_signals['t'][0]
ia = emitter_signals['ia']

first_filter = AntiAliasingFilter(period=original_period, signal=ia, b=1.599e3, c=1.279e6)
first_filter.apply_filter()
filtered_ia = first_filter.filtered_signal

md = int(1e-3 / original_period)
resampled_ia = resample(filtered_ia, md)
new_time_points = list(emitter_signals['t'][::md])
new_period = new_time_points[-1] - new_time_points[-2]

phasor_ia = PhasorEstimator(resampled_ia, sample_rate=16)
phasor_ia.estimate()

figure, axis = plt.subplots(2, 1)
axis[0].plot(emitter_signals['t'], ia, label='ia')
axis[0].scatter(new_time_points, resampled_ia, label=" ia pós antialiasing (reamostrado)", color='red', s=3)
axis[0].plot(new_time_points, phasor_ia.amplitude[:2301], label='amplitude')
axis[0].legend()
axis[1].plot(new_time_points, phasor_ia.phase[:2301], label='phase')
axis[1].legend()
plt.show()
