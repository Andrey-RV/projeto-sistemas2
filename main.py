import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iec import Iec

FUNDAMENTAL_PERIOD = 1 / 60
DESIRED_SAMPLES_PER_CYCLE = 16
R = 0.0246 * 250
XL = 0.3219 * 250

signals = {}
for fault in range(1, 3):
    for bus in range(1, 3):
        signals[f'bus{bus + 1}_fault{fault}'] = pd.read_csv(f"./Registros/Registro{fault}/1Reg{bus}.dat", delimiter='\s+',
                                                            names=['step', 't', 'Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic'])

sampling_period = signals['bus2_fault1']['t'][1] - signals['bus2_fault1']['t'][0]
md = FUNDAMENTAL_PERIOD / (DESIRED_SAMPLES_PER_CYCLE * sampling_period)

phase_timed_adjust_current = 3
phase_insta_adjust_current = {'c': 5.03, 'b': 5.49, 'a': 6.05, 'b\'': 11.68, 'c\'': 14.71, 'd\'': 19.87}
neutral_timed_adjust_current = 0.5
neutral_insta_adjust_current = {'c': 1.01, 'b': 1.10, 'a': 1.21, 'b\'': 2.34, 'c\'': 2.94, 'd\'': 3.97}


def create_iec(bus, fault):
    iec = Iec(
        va=signals[f'bus{bus}_fault{fault}']['Va'],
        vb=signals[f'bus{bus}_fault{fault}']['Vb'],
        vc=signals[f'bus{bus}_fault{fault}']['Vc'],
        ia=signals[f'bus{bus}_fault{fault}']['Ia'],
        ib=signals[f'bus{bus}_fault{fault}']['Ib'],
        ic=signals[f'bus{bus}_fault{fault}']['Ic'],
        t=signals[f'bus{bus}_fault{fault}']['t'],
        sampling_period=sampling_period,
        b=1.599e3, c=1.279e6, md=md, R=R, XL=XL,
        estimator_samples_per_cycle=16, RTC=1
    )
    return iec


def add_67(iec: Iec, gamma_phase, gamma_neutral):
    iec.add_67(type='phase', gamma=gamma_phase, timed_adjust_current=phase_timed_adjust_current,
               insta_adjust_current=phase_insta_adjust_current, curve='IEEE_moderately_inverse')
    iec.add_67(type='neutral', gamma=gamma_neutral, timed_adjust_current=neutral_timed_adjust_current,
               insta_adjust_current=neutral_insta_adjust_current, curve='IEEE_moderately_inverse')


def plot_part1(iec: Iec, title, resample_limit=0.7):
    resampled_time = [iec.time[i] for i in range(len(iec.time)) if iec.time[i] < resample_limit]
    figure, axis = plt.subplots(9, 1, figsize=(10, 15))

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i].plot(iec.time, iec.logical_state_51F[phase], label=f'51{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for i, phase in enumerate(['ia', 'ib', 'ic']):
        axis[i + 3].plot(iec.time, iec.logical_state_50F[phase], label=f'50{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for i, phase in enumerate(['a', 'b', 'c']):
        axis[i + 6].plot(resampled_time, iec.trip_permission_32F[phase][:len(resampled_time)], label=f'32{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    for ax in figure.get_axes():
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize='small', handlelength=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])

    plt.suptitle(title)
    plt.show()


def plot_part2(iec: Iec, title, resample_limit=0.7):
    resampled_time = [iec.time[i] for i in range(len(iec.time)) if iec.time[i] < resample_limit]
    figure, axis = plt.subplots(8, 1, figsize=(10, 15))

    axis[0].plot(iec.time, iec.logical_state_51N, label='51N', linewidth=2, color='black')
    axis[1].plot(iec.time, iec.logical_state_50N, label='50N', linewidth=2, color='black')
    axis[2].plot(resampled_time, iec.trip_permission_32N[:len(resampled_time)], label='32N', linewidth=2, color='black')

    for i, phase in enumerate(['a', 'b', 'c']):
        axis[i + 3].plot(iec.time, iec.trip_permission_67F[phase], label=f'67{chr(65+i)}', linewidth=2, color=['black', 'blue', 'red'][i])

    axis[6].plot(iec.time, iec.trip_permission_67N, label='67N', linewidth=2, color='black')
    axis[7].plot(iec.time, iec.trip_signal, label='Trip signal', linewidth=2, color='black')

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


for bus in range(2, 4):
    gamma_phase = 0.069 if bus == 2 else 0.173
    gamma_neutral = 0.240 if bus == 2 else 0.312

    for fault in range(1, 3):
        iec = create_iec(bus, fault)
        add_67(iec, gamma_phase, gamma_neutral)
        plot_part1(iec, f'Bus {bus} Fault {fault}')
        plot_part2(iec, f'Bus {bus} Fault {fault}')
