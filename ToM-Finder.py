import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# -------- USER INPUT --------

file = "file.csv"
T0 =  00000  # rough reference minimum
P = 0.0

window_fraction = 0.1   # window size around predicted minima

# ----------------------------

# load data
data = pd.read_csv(file)

time = np.array(data['time'])
flux = np.array(data['mag'])


# parabola model
def parabola(t, a, b, c):
    return a*t**2 + b*t + c


# find minima function
def find_minimum(t_segment, f_segment):

    t_shift = t_segment - np.mean(t_segment)

    popt, _ = curve_fit(parabola, t_shift, f_segment)

    a, b, c = popt

    t_min_shift = -b/(2*a)

    t_min = t_min_shift + np.mean(t_segment)

    return t_min


# find predicted epochs
E_min = int((min(time) - T0)/P) - 2
E_max = int((max(time) - T0)/P) + 2

epochs = np.arange(E_min, E_max + 1)

minima_times = []


for E in epochs:

    T_pred = T0 + E*P

    window = window_fraction * P

    mask = (time > T_pred - window) & (time < T_pred + window)

    t_seg = time[mask]
    f_seg = flux[mask]

    if len(t_seg) < 10:
        continue

    try:
        Tmin = find_minimum(t_seg, f_seg)

        minima_times.append([E, Tmin])

    except:
        continue


minima_df = pd.DataFrame(minima_times, columns=["Epoch", "Tmin"])

minima_df.to_csv("times_of_minima.csv", index=False)

print(minima_df)