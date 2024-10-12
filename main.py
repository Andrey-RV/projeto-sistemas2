import pandas as pd
import matplotlib.pyplot as plt
from iec import Iec


R = 0.0246 * 200  # 200 km de linha de transmissão com resistência de 0.0246 ohm/km.
XL = 0.3219 * 200  # 200 km de linha de transmissão com reatância de 0.3219 ohm/km.


emitter_signals = pd.read_csv("./Atividade_01/1Reg1.dat", delimiter='\s+',
                              names=['1', 't', '3', '4', '5', '6', 'va', 'vb', 'vc',
                                     '10', 'ia', 'ib', 'ic', '14', '15', '16', '17', '18'])

original_sampling_period = emitter_signals['t'][1] - emitter_signals['t'][0]
t = emitter_signals['t']
va = emitter_signals['va']
vb = emitter_signals['vb']
vc = emitter_signals['vc']
ia = emitter_signals['ia']
ib = emitter_signals['ib']
ic = emitter_signals['ic']

md = 1e-3 / original_sampling_period  # Fator de subamostragem.

overcurrent_relay = Iec(va=va, vb=vb, vc=vc, ia=ia, ib=ib, ic=ic, t=t,
                        b=1.599e3, c=1.279e6, sampling_period=original_sampling_period,
                        R=R, XL=XL, md=md, phasor_estimator_samples_per_cycle=16)

# The code above is the main.py file that uses the Iec class from iec.py. The Iec class is responsible for estimating the phasors of the signals va, vb, vc, ia, ib, and ic. The phasor estimation process is done by applying an anti-aliasing filter, resampling the signals, applying a mimic filter, and estimating the phasors. The phasor estimation is done by the PhasorEstimator class. The main.py file reads a CSV file with the signals and creates an instance of the Iec class to estimate
