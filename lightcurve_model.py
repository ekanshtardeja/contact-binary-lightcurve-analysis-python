import phoebe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

b = phoebe.default_binary()

data = pd.read_csv("file.csv")

time = data['time'].values
flux = data['mag'].values

# mag → flux
flux = 10**(-0.4 * flux)
flux = flux / np.max(flux)

# your inputs
T0 = 3543.97865686
P = 0.30690791177295174

# phase
phase = ((time - T0) / P) % 1

# sort
idx = np.argsort(phase)
phase = phase[idx]
flux = flux[idx]

b.add_dataset('lc',
              times=phase,
              fluxes=flux,
              dataset='lc01')


b['period@binary'] = 1
b['incl@binary'] = 90                    
b['q@binary'] = 1

b['teff@primary'] = 5000
b['teff@secondary'] = 5500



b['ntriangles@primary'] = 100
b['ntriangles@secondary'] = 100
b.run_compute()

model_phase = b['times@model'].value
model_flux = b['fluxes@model'].value

model_flux = model_flux / np.max(model_flux)

plt.figure(figsize=(8,5))

plt.scatter(phase, flux, s=5, color='black', label='Observed')
plt.plot(model_phase, model_flux, color='red', label='Model')

plt.xlabel("Phase")
plt.ylabel("Normalized Flux")
plt.legend()

plt.show()