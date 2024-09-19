import numpy as np 

import matplotlib
matplotlib.use('Agg')

from numpy import pi, loadtxt, array, zeros, ones, size, linspace, sort
from numpy import exp, cos 
from numpy.random import randint, randn
import glob 

import matplotlib.pylab as plt
from matplotlib.pylab import figure, axis, axes, plot, imshow, hist, histogram, contour, contourf, clim
from matplotlib.pylab import xlabel, ylabel, title, annotate, text
from matplotlib.pylab import xticks, yticks, tick_params, ticklabel_format
from matplotlib.pylab import show, savefig, close, colorbar
from matplotlib.figure import Figure

from datetime import datetime


#------
def grid_max(nuvec, nudot_vec, pcos_vec, amp_vec, phase_vec, Dvec):
	max = np.argmax(Dvec)
	max_ind = np.unravel_index(max, Dvec.shape)
	max_val = Dvec[max_ind]
	return(max_ind, max_val)

	#finds the combination of parameter values that correlates with the highest det stat
	#holds both the parameter value and the index for later use

#------- 
search_res = np.load('Dvec3.npz', allow_pickle = True)

Dvec3_plot = search_res['Dvec3']
nu_grid = search_res['p']
nudot_grid = search_res['pdot']
pcos_grid = search_res['pcos']
amp_grid = search_res['amp']
phase_grid = search_res['phase']

#pulls the search grids for each parameter

#------
max_ind, max_val = grid_max(nu_grid, nudot_grid, pcos_grid, amp_grid, phase_grid, Dvec3_plot)
mesh = np.meshgrid(nu_grid, nudot_grid, pcos_grid, amp_grid, phase_grid)

#---
fig, axs = plt.subplots(figsize = (9,8), nrows = 5, ncols = 5)

plt.suptitle('FRB Parameter Detections')

plt.subplots_adjust(right = 0.95, bottom = 0.1, top = 0.85, wspace = 0.6, hspace = 0.6)


level = np.linspace(0,1,100)


for i in range(0,4):
	for j in range(0,5):
		if i < j: 
			axs[i, j].axis('off')

im=axs[4,0].contourf(phase_grid, nu_grid, Dvec3_plot[:,max_ind[1],max_ind[2],max_ind[3],:])

axs[4,1].contourf(amp_grid, nu_grid, Dvec3_plot[:,max_ind[1], max_ind[2], :, max_ind[4]])

axs[4,2].contourf(pcos_grid, nu_grid, Dvec3_plot[:,max_ind[1], :, max_ind[3], max_ind[4]])

axs[4,3].contourf(nudot_grid, nu_grid, Dvec3_plot[:,:,max_ind[2],max_ind[3],max_ind[4]])

axs[4,4].plot(nu_grid, Dvec3_plot[:,max_ind[1],max_ind[2],max_ind[3],max_ind[4]])


axs[3,0].contourf(phase_grid, nudot_grid, Dvec3_plot[max_ind[0],:,max_ind[2],max_ind[3],:])

axs[3,1].contourf(amp_grid, nudot_grid, Dvec3_plot[max_ind[0],:,max_ind[2], :, max_ind[4]])

axs[3,2].contourf(pcos_grid, nudot_grid, Dvec3_plot[max_ind[0],:, :, max_ind[3], max_ind[4]])

axs[3,3].plot(nudot_grid, Dvec3_plot[max_ind[0],:, max_ind[2], max_ind[3], max_ind[4]])


axs[2,0].contourf(phase_grid, pcos_grid, Dvec3_plot[max_ind[0],max_ind[1],:,max_ind[3],:])

axs[2,1].contourf(amp_grid, pcos_grid, Dvec3_plot[max_ind[0],max_ind[1],:,:,max_ind[4]])

axs[2,2].plot(pcos_grid, Dvec3_plot[max_ind[0],max_ind[1],:,max_ind[3],max_ind[4]])


axs[1,0].contourf(phase_grid, amp_grid, Dvec3_plot[max_ind[0],max_ind[1], max_ind[2], :,:])

axs[1,1].plot(amp_grid, Dvec3_plot[max_ind[0],max_ind[1], max_ind[2],:, max_ind[4]])


axs[0,0].plot(phase_grid, Dvec3_plot[max_ind[0],max_ind[1], max_ind[2], max_ind[3] ,:])

#generate a corner plot to show the detection statistic values across every combination of search grids
	#use the max_ind for the other 3 parameters to pick a 'slice' of the 5D grid at which to plot the other two search grids 
#includes a 2 dimensional plot at the top of each column for easy identification



axs[4,0].set_xlabel('phase')
axs[4,1].set_xlabel('amplitude')
axs[4,2].set_xlabel('cosine period')
axs[4,3].set_xlabel('spindown')
axs[4,4].set_xlabel('period')

axs[4,0].set_ylabel('period')
axs[3,0].set_ylabel('spindown')
axs[2,0].set_ylabel('cosine period')
axs[1,0].set_ylabel('amplitude')
axs[0,0].set_ylabel('phase')

for x in range(0,4):
	for y in range(0,4):
		axs[x,y].ticklabel_format(axis = 'both', style = 'scientific', scilimits = (0,0))

#cax = plt.axes((0.85, 0.1, 0.05, 0.8))
plt.colorbar(im, ax=axs,orientation='vertical')
fig.set_size_inches(12,9)

fig.savefig('frb_psearch_cornerplot.jpg', dpi = 300)


