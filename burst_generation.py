
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, arange, zeros, append, savetxt, cos
from numpy import random, loadtxt

from matplotlib.pylab import plot, show, figure, xlabel, ylabel, title, tick_params

import glob

infile = 'parameters.txt'

params = (loadtxt(infile, unpack=True)).astype(float)

t_0 = params[0]
t_obs = params[1]
p = params[2]
pdot = params[3]
pcos = params[4]
amp = params[5]
phase = params[6]


duty_2 = params[7]
duty_1 = 1 - duty_2

n = params[8]

#initialize all parameters from an input file that's editable by user

time_arr = arange(t_0, t_obs, .0001)
#set up an array of observation times to look through
#step size should vary with the complexity of your parameters
	#i.e. smaller step size for more complex and smaller parameters


toas = []

dphi_err = 0.00005

for t in time_arr: 
	phi = p*t + 0.5*pdot*t**2 + amp*cos(pcos*t+phase)
	dphi = phi - (phi+0.5).astype(int)
	if abs(dphi) < dphi_err:
		noise = (random.normal(0, n, 1)) * nu
		prob = random.choice(2, 1, p = [duty_1, duty_2])
		if prob == 0:
			pass
		elif prob == 1:
			t_fin = t*prob + noise
			toas.append(t_fin[0])

#using the phase residuals vs pre-set error to check which observation times would correlate with the given parameters
#adding in noise & missed detections to mimic real data

np.savez('sim_bursts_1.npz', toas = toas, params = params)


