#File frb121102_fast-periodicity_search_taite_feb20.py

import numpy as np 
import time

from numpy import pi, loadtxt, savetxt, array, zeros, ones, size, linspace, sort
from numpy import exp, cos 
from numpy.random import randint, randn
import glob 

from matplotlib.pylab import figure, axis, axes, plot, imshow, hist, histogram, contour, contourf
from matplotlib.pylab import xlabel, ylabel, title, annotate, text
from matplotlib.pylab import xticks, yticks, tick_params
from matplotlib.pylab import show, savefig, close, colorbar

from datetime import datetime


def phimod1 (t, p, phi0=0):
	phi = phi0 + p*t
	dphi = phi - (phi+0.5).astype(int)
	return phi, dphi 

def phimod2 (t, p, pdot, phi0=0):
	phi = phi0 + p*t + 0.5*pdot*t**2
	dphi = phi - (phi+0.5).astype(int)
	return phi, dphi

def phimod3 (t, p, pdot, p_cos, amp, phase):
	phi = p*t + 0.5*pdot*t**2 + amp*cos(p_cos*t + phase)
	dphi = phi - (phi+0.5).astype(int)
	return phi, dphi

def Detect(dphi):
	D = sum(exp(2*pi*1j*dphi)) / size(dphi)
	Dmag = abs(D)
	return Dmag

	#calculates phase residuals

def calc_detect1(dt_sec, nuvec):
	Dvec = zeros(size(nuvec))
	for n, p in enumerate(nuvec):
		phivec, dphivec = phimod1(dt_sec, p)
		Dmag = Detect(dphivec)
		Dvec[n] = Dmag 
	return Dvec

	#calculate detection statistic for each possible period value 

def calc_detect2(dt_sec, pvec, pdot_vec):
	Dvec2 = np.zeros((len(pdot_vec),len(pvec)))
	for x, p in enumerate(pvec):
		for y, pdot in enumerate(pdot_vec):
			phi, dphi = phimod2(dt_sec, p, pdot) 
			D_stat = Detect(dphi)
			Dvec2[y,x] = D_stat
	return Dvec2

	#calculate dstat for each combination of period and spindown values


def calc_detect3(dt_sec, pvec, pdot_vec, p_cos_vec, amp_vec, phase_vec):
	Dvec3 = np.zeros((len(pvec),len(pdot_vec), len(p_cos_vec), len(amp_vec), len(phase_vec)))
	i = 0
	f = open('progress.txt', 'w')
	f.close()
	for x, p in enumerate(nuvec):
		prog = i/len(pvec)*100
		f = open('progress.txt', 'a')
		f.write('progess ' + str(prog) + '%' + '\n')
		print('progess ' + str(prog) + '%')
		f.close()
		i += 1
		for y, pdot in enumerate(pdot_vec):
			for z, p_cos in enumerate(p_cos_vec):
				for a, amp in enumerate(amp_vec):
					for p, phase in enumerate(phase_vec):
						phi, dphi = phimod3(dt_sec, p, pdot, p_cos, amp, phase) 
						D_stat = Detect(dphi)
						if D_stat > 1:
							D_stat = D_stat - 1
						Dvec3[x,y,z,a,p] = D_stat
	f.close()
	return Dvec3

	#calculates dstat for every combination of possible period, spindown, sinusoidal period, amplitude, and phase values 
	#also creates & updates a file to track progress since this can take a while


#-----------------------------------------------------------------------------
day = 86400 
time = 100

infile = 'sim_bursts_1.npz'

bursts = np.load(infile)
toas = bursts['toas']

bursts.close()

#loads in data


#------------------------------------
#plot FRB events across time

'''fig = figure()
ax = fig.add_subplot(111)
plot(toas, 'k.', ms = 5)
xlabel(r'$\rm Barycentric \ Time \ \ \ (s))$', fontsize=15)
ylabel(r'$\rm Burst \ events$', fontsize = 15)
title('Burst events of Simulated data')
tick_params(axis='y', labelleft=False)
show()'''


#-----------------------------------
numin = 1
numax = 10
d_nu = 0.01

nudot_min = -1e-5
nudot_max = 1e-5
d_nudot = 1e-6

pcos_min = -50
pcos_max = 50
d_pcos = 1

amp_min = 0
amp_max = 1
d_amp = .1


phase_min = 0
phase_max = 1
d_phase = 0.1

N_nu = int((numax - numin)/ d_nu) + 1
N_nudot = int((nudot_max - nudot_min)/ d_nudot) + 1
N_pcos = int((pcos_max - pcos_min) / d_pcos) + 1
N_amp = int((amp_max - amp_min) / d_amp) + 1
N_phase = int((phase_max - phase_min) / d_phase) + 1


nu_grid = linspace(numin, numax, N_nu, endpoint = True)
nudot_grid = linspace(nudot_min, nudot_max, N_nudot, endpoint = True)
pcos_grid = linspace(pcos_min, pcos_max, N_pcos, endpoint = True)
amp_grid = linspace(amp_min, amp_max, N_amp, endpoint = True)
phase_grid = linspace(phase_min, phase_max, N_phase, endpoint = True)

#set up grid of values to search across for each parameter


Dvec3_search = calc_detect3(toas, nu_grid, nudot_grid, pcos_grid, amp_grid, phase_grid)


np.savez('Dvec3.npz', Dvec3 = Dvec3_search, nu = nu_grid, nudot = nudot_grid, pcos = pcos_grid, amp = amp_grid, phase = phase_grid)

#generate a 5D grid of detection statistic values for each combination of possible parameter values 



