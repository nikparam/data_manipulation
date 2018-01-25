from functools import partial, reduce
import numpy as np
import matplotlib.pyplot as plt

def compose(*func):
	init, *rest = reversed( func )
	return lambda *args: reduce( lambda a, f: f(a), rest, init(*args) )

lmap = compose( list, map )

'''
Data
'''

shift_TMS = 31.3240
#shifts_exp = [ 31.1662, 27.4712, 27.4712, 30.3121, 30.0346, 30.0346 ]
shifts_exp = [ 31.1662, 27.4712, 27.4712, 30.2171, 30.2171, 30.2171]
n = len(shifts_exp)
SSC = [ [ 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 ],\
	[ 0.00000, 0.00000, 5.08944, 5.08944, 5.08944 ],\
	[ 0.00000, 0.00000, 5.08944, 5.08944, 5.08944 ],\
	[ 0.00000, 5.08944, 5.08944, 0.00000, 0.00000 ],\
	[ 0.00000, 5.08944, 5.08944, 0.00000, 0.00000 ],\
	[ 0.00000, 5.08944, 5.08944, 0.00000, 0.00000 ] ]

#SSC = [ [ 0.804711,  0.804711, 2.04605, -0.891149, -0.891149],\
#       [ 0.804711, -8.929959, 1.51643,   9.90495,   3.84695],\
#      	[ 0.804711, -8.929959, 1.51643,   3.84695,   9.90495],\
#      	[  2.04605,   1.51643, 1.51643,  -13.1501,  -13.1501],\
#       [-0.891149,   9.90495, 3.84695,  -13.1501,  -12.4454],\
#      	[-0.891149,   3.84695, 9.90495,  -13.1501,  -12.4454] ]

SSC_ppm = lmap( lambda x: lmap( lambda y: y/shift_TMS, x ), SSC )

print(SSC_ppm)

width = 0.01

'''
General functions
'''

corr_shifts = partial( lambda a, x: a - x, shift_TMS )
corr_centers = lmap( corr_shifts, shifts_exp )

laplace_gen = lambda gamma, x0, x: 0.5 * gamma / ( np.pi * ( (x - x0)**2 + 0.25 * gamma**2)) 
laplace = partial( laplace_gen, width )
laplace_centered = lambda x: partial(laplace, x)

spectra = lambda func_list, y, x: reduce( lambda a, f: a + f(x) / y, \
						      func_list[1:], \
						      func_list[0](x) / y )

shifts = np.linspace( 0, 5, 5000)

'''
no SSC
'''

func = lmap( laplace_centered, corr_centers )

'''
SSC
'''

ssc_p = lmap( lambda x, y: lmap( lambda z: partial(x,z), y ), [ lambda x0, x: x + x0] * len(shifts_exp), SSC_ppm )
ssc_m = lmap( lambda x, y: lmap( lambda z: partial(x,z), y ), [ lambda x0, x: x - x0] * len(shifts_exp), SSC_ppm )


ssc_corr_shifts = []

for _ in range( n ):
	for i in range( 2**(n-1) ):
		s = bin( i )[2:].zfill(n-1)
		hv = lambda x: x
		for k, l in enumerate(s):
			if l == '0': hv = compose( ssc_m[_][k], hv )
			else:	     hv = compose( ssc_p[_][k], hv )
		print( i+1, s, hv(corr_centers[_]) )

		ssc_corr_shifts.append( hv( corr_centers[_] ) )
	print()


func_ssc = lmap( laplace_centered, ssc_corr_shifts )

'''
Plot
'''

fig=plt.figure(figsize=(10,8),dpi=100,facecolor='w',edgecolor='k')
ax=fig.add_axes([0.15,0.15,0.7,0.7])

#ax.plot(shifts, partial(spectra,func,1)(shifts))
ax.plot(shifts, partial(spectra,func_ssc,2**(n-1))(shifts))

ax.set_xlim([0,5])
ax.invert_xaxis()
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.axes.get_yaxis().set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$\delta,\ ppm$',fontsize=25)
ax.xaxis.set_label_coords(.5,-0.055)
plt.xticks(fontsize=15)
plt.show()

