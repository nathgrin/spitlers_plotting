


def plot_progression(x,y,dat,repeat_on_x=True,labels={},\
                      kwargs_axmain={},kwargs_axside={},kwargs_trends={},kwargs_highlight={},kwargs_infotxt={},\
                      fontsize_ticks=10,verbose=True,\
                      kwargs_saveshow={}):
	""" A plot to show
		1) A (main) image     y vs x with color dat
		2) A top   lineplot dat vs x for each y (colored by y)
		3) A right lineplot dat vs y for each x (colored by x)
		The colorbar in the top-right has two scales; for the top (top) and right (bottom) panels
	
	Parameters
	----------
	x,y : array_like
	  have to be evenly spaced
	dat : array_like, shape (y,x)
	  data values
	
	repeat_on_x : bool (default: True)
	  Repeat the image shifted in x (by max(x)-min(x) + abs(x[1]-x[0])) and y (by abs(y[1]-y[0]))
	  This is to see progression e.g., as one year passes into the next
	
	labels : dict
		Optional entries:
			x,y,c : string (default: None)
			fontsize : float (default: 14)
	
	kwargs_axmain : dict
	  Kwargs for the main image
		Optional entries:
			cmap : string, see ~matplotlib.colors.Colormap (default: viridis)
			vmin,vmax : float, min resp. max for cbar (default: None)
	
	kwargs_axside : dict
	  Kwargs for the top and right panels
		Optional entries:
			cmap : string, see ~matplotlib.colors.Colormap (default: inferno_spitler)
			  if inferno_spitler is not available, use cividis
			  (inferno_spitler is inferno, cut to exclude the lightest bit)
			vmin_top,vmax_top : float, for cbar (default: min(y),max(y))
			vmin_right,vmax_right : float, for cbar (default: min(x),max(x))
			lw : float (default: 0.5)
			alpha : float (default: 1.)
			zorder : float (default: 10.)
			
	kwargs_trends : dict
	  To plot statistics (mean, median, etc) for the trends in the top and right panel
		  i.e., as function of x resp. y
		Optional entries:
			Set these to True to plot:
			  mean : bool (default: True)
			  median : bool (default: False)
			  std : bool (default: True)
			  quantiles : bool (default: False)
			use entry_linestyle, which will be passed to plt.plot
			  Possible linestyle kwargs: _c,_ls, _lw, _alpha, _zorder
			  e.g., 'mean_ls' : '--'
			  all default to mean_*, if not set seperately
			quantiles uses _q : list of float or flat (default [0.3,0.7])
			  A list or single value of quantiles to be plotted
			ignore_nans : bool (default: True)
			  if ignore_nans, use e.g., np.nanmean in stead of np.mean etc.
	
	kwargs_highlight : dict
	  To highlight certain rows/columns on the top/right panels
		Optional entries:
			top : list (default: None)
			right : list (default: None)
			  list of x/y values to be highlighted
			  as of now, no check is made whether they actually exist
			Set c, lw, ls, alpha, zorder to overwrite the _axside defaults
			c : mpl color (default: None)
			  defaults to coloring by x/y value as normal
			alpha : float (default: 1.)
			zorder : float (default: 20.)
			lw : float (default: 1.5)
	
	kwargs_infotxt : dict
	  To place text in bottom-right corner as extra information
		Optional entries:
			txt : str (default: None)
			  the actual text
			fontsize: float (default: labels['fontsize'])
			Other kwargs can be added and will be directly passed to plt.ax.txt()
	
	fontsize_ticks : float (default: 10)
	
	verbose : bool (default: True)
	  Print some info on the data
	
	
	kwargs_saveshow : dict
	  Parameters for saving/showing the figure
		Optional entries:
			show : bool (default: True)
			save : bool (default: False)
			  Do the save or show?
			path : string (default: "figures/")
			name : string (default: None)
			  if None, defaults to the name of the current function
			name_extra : string (default: "")
			ext : string (default: ".pdf")
			  extension
			Note:
				resulting filename = path +  name + name_extra + ext
				(default: "figures/function_name.pdf")
	
	
	Returns
	----------
	dictionary with keys ax_main, ax_top, ax_right
	  in case you want to overplot something
	
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
	import matplotlib.gridspec as gridspec
	
	from misc_func import handle_colorbar
	from misc_func import set_kwargdict_defaults,dict_set_default
	
	
	### Handle kwargs
	
	# labels
	defaults_labels = { 
	    'x': None,
	    'y': None,
	    'c': None,
	    'fontsize': 14
	    }
	labels = set_kwargdict_defaults( labels, defaults_labels, "labels" )
	
	# axes
	defaults_kwargs_axmain = {
	    'cmap': 'viridis',
	    'vmin': None,
	    'vmax': None
	    }
	kwargs_axmain = set_kwargdict_defaults( kwargs_axmain, defaults_kwargs_axmain, "kwargs_axmain" )
	
	defaults_kwargs_axside = {
	    'cmap': 'inferno_spitler' if 'inferno_spitler' in plt.colormaps() else 'cividis',
	    'vmin_top': min(y),
	    'vmax_top': max(y),
	    'vmin_right': min(x),
	    'vmax_right': max(x),
	    'lw': 0.5,
	    'alpha': 1.,
	    'zorder': 10.
	    }
	kwargs_axside = set_kwargdict_defaults( kwargs_axside, defaults_kwargs_axside, "kwargs_axside" )
	
	# Trends
	defaults_kwargs_trends = {
	    'ignore_nans': True,
	    
	    'mean': True,
	    'mean_ls': '-',
	    'mean_lw': 0.8,
	    'mean_alpha': 1.,
	    'mean_c': 'lime',
	    'mean_zorder': 20.,
	    
	    'median': False,
	    'median_ls': '-.',
	    
	    'std': True,
	    'std_ls': '--',
	    
	    'quantiles': False,
	    'quantiles_q': [0.3,0.7],
	    'quantiles_ls': ':'
	    }
	for stat in ['mean','median','std','quantiles']:
		dict_set_default( defaults_kwargs_trends, stat+'_lw',     defaults_kwargs_trends['mean_lw'])
		dict_set_default( defaults_kwargs_trends, stat+'_alpha',  defaults_kwargs_trends['mean_alpha'])
		dict_set_default( defaults_kwargs_trends, stat+'_c',      defaults_kwargs_trends['mean_c'])
		dict_set_default( defaults_kwargs_trends, stat+'_zorder', defaults_kwargs_trends['mean_zorder'])
	kwargs_trends = set_kwargdict_defaults( kwargs_trends, defaults_kwargs_trends, "kwargs_trends" )
	
	if type(kwargs_trends['quantiles_q']) == float: # Other type (and range) checking will be done by np.quantile
		kwargs_trends['quantiles_q'] = [kwargs_trends['quantiles_q']]
		
	
	# Highlight in the top/right panel
	defaults_kwargs_highlight = {
	    'top':   None,
	    'right': None,
	    'c': None,
	    'lw': 1.5,
	    'ls': '-',
	    'alpha': 1.,
	    'zorder': 20.
	    }
	kwargs_highlight = set_kwargdict_defaults( kwargs_highlight, defaults_kwargs_highlight, "kwargs_highlight" )
	
	if ((kwargs_highlight['top'] is not None) and (type(kwargs_highlight['top']) != list)) \
	  or ((kwargs_highlight['right'] is not None) and (type(kwargs_highlight['right']) != list)):
		raise ValueError("kwargs_highlight: 'top' and 'right' must be None or list")
	
	
	# infotxt (in bottom-right)
	defaults_kwargs_infotxt = {
	    'txt': None,
	    'fontsize': labels['fontsize']
	    }
	kwargs_infotxt = set_kwargdict_defaults( kwargs_infotxt, defaults_kwargs_infotxt, "kwargs_infotxt" )
	
	# aux
	#~ fontsize_ticks = kwargs.get('fontsize_ticks',10)
	
	#~ verbose = kwargs.get('verbose', True)
	
	
	# Save/Show
	defaults_kwargs_saveshow = {
	    'show': True,
	    'save': False,
	    'path': "figures/",
	    'name': None,
	    'name_extra': "",
	    'ext': ".pdf",
	    }
	kwargs_saveshow = set_kwargdict_defaults( kwargs_saveshow, defaults_kwargs_saveshow, "kwargs_saveshow" )
	
	
	### Data check
	
	if verbose:
		print(' > Progression plot')
		print('   len(x): %i, len(y): %i, dat.shape: %s'%(len(x),len(y),str(dat.shape)) )
		#~ input()
	
	if np.sum(abs(np.diff(np.diff(x)))) > 1e-10 or np.sum(abs(np.diff(np.diff(y)))) > 1e-10:
		raise Exception("x and y have to be evenly spaced arrays")
		
	if len(dat) != len(y) and np.sum([len(xi) for xi in dat])/len(dat) != len(x):
		raise Exception("dat has to be rectangular with shape (y,x)")
		
	if repeat_on_x:
		
		x_offset = abs(x[-1]-x[0]) + abs(x[1]-x[0])
		y_offset = abs(y[1]-y[0])
		
		
	### Setup
	fig = plt.figure()
	
	
	rat = (5,13,3)
	height_ratios = (rat[0]*0.425,rat[0]*0.15,rat[0]*0.425,rat[1],rat[2]*0.7,rat[2]*0.3,rat[2]*0.00)
	gs = gridspec.GridSpec(ncols=3,nrows=7, figure=fig, height_ratios=height_ratios, width_ratios=(8, 0.2, 2),wspace=0.,hspace=0.)
	ax_top   = fig.add_subplot(gs[:3,0])
	ax_main  = fig.add_subplot(gs[3,0],sharex=ax_top)
	cax_main = fig.add_subplot(gs[5:,0])
	ax_right = fig.add_subplot(gs[3,1:],sharey=ax_main)
	ax_infotxt = fig.add_subplot(gs[5:,1:])
	cax_other= fig.add_subplot(gs[1,2])
	
	
	
	### Content
	## Main img
	# repeat to show next year continue
	cmap = kwargs_axmain['cmap']
	vmin,vmax = kwargs_axmain['vmin'],kwargs_axmain['vmax']
	im = ax_main.pcolormesh(x,y,dat,cmap=cmap,shading='auto',vmin=vmin,vmax=vmax)
	if repeat_on_x:
		im = ax_main.pcolormesh(x+x_offset,y-y_offset,dat,cmap=cmap,shading='auto',vmin=vmin,vmax=vmax)
	
	cb_main = fig.colorbar(im,cax=cax_main,orientation='horizontal')
	
	## Top and right
	# Some prep
	cmap_topright = kwargs_axside['cmap']
	
	do_highlight_top   = kwargs_highlight['top']   is not None
	do_highlight_right = kwargs_highlight['right'] is not None
	change_highlight_c = kwargs_highlight['c'] is not None
	lw_def,alpha_def = kwargs_axside['lw'],kwargs_axside['alpha']
	ls_def,zorder_def = '-',10.
	lw_hig,alpha_hig = kwargs_highlight['lw'],kwargs_highlight['alpha']
	ls_hig,zorder_hig = kwargs_highlight['ls'],kwargs_highlight['zorder']
	
	# Top first
	cbnorm_y,cbax_y = handle_colorbar(kwargs_axside['vmin_top'],kwargs_axside['vmax_top'],cmap=cmap_topright)
	
	for i,row in enumerate(dat):
		if do_highlight_top and (y[i] in kwargs_highlight['top']):
			c = kwargs_highlight['c'] if change_highlight_c else  cbnorm_y(y[i])
			lw,alpha,ls,zorder = lw_hig,alpha_hig,ls_hig,zorder_hig
		else:
			c,lw,alpha,ls,zorder = cbnorm_y(y[i]),lw_def,alpha_def,ls_def,zorder_def
		
		ax_top.plot(x,row,lw=lw,alpha=alpha,c=c,ls=ls,zorder=zorder)
		
		if repeat_on_x:
			ax_top.plot(x+x_offset,row,lw=lw,alpha=alpha,c=cbnorm_y(y[i]))
		#~ print(row)
	# Right next
	cbnorm_x,cbax_x = handle_colorbar(kwargs_axside['vmin_right'],kwargs_axside['vmax_right'],cmap=cmap_topright)
	
	for i,col in enumerate(dat.T):
		if do_highlight_right and (x[i] in kwargs_highlight['right']):
			c = kwargs_highlight['c'] if change_highlight_c else  cbnorm_x(x[i])
			lw,alpha,ls,zorder = lw_hig,alpha_hig,ls_hig,zorder_hig
		else:
			c,lw,alpha,ls,zorder = cbnorm_x(x[i]),lw_def,alpha_def,ls_def,zorder_def
		
		ax_right.plot(col,y,lw=lw,alpha=alpha,c=c,ls=ls,zorder=zorder)
		#~ print(row)
		
	# Mean and/or medium
	def xwise_ywise_stat(dat,func,**kwargs):
		return func(dat,**kwargs),func(dat.T,**kwargs)
	stat_to_func = {'mean': (np.mean,np.nanmean),'median': (np.median,np.nanmedian),'std': (np.std,np.nanstd),'quantiles': (np.quantile,np.nanquantile)}
	stat_func_kwargs = {'axis':0}
	
	kwargs_plottrend = {'ls':'--','lw':0.8,'alpha':1.,'c':'r','zorder':20.}
	for stat in ['mean','median']:
		if kwargs_trends[stat]:
			func = stat_to_func[stat][kwargs_trends['ignore_nans']]
			xwise,ywise = xwise_ywise_stat(dat,func,**stat_func_kwargs)
			
			for entry in kwargs_plottrend.keys(): kwargs_plottrend[entry] = kwargs_trends[stat+'_'+entry]
			
			ax_top.plot(x,xwise,**kwargs_plottrend)
			if repeat_on_x:
				ax_top.plot(x+x_offset,xwise,**kwargs_plottrend)
			ax_right.plot(ywise,y,**kwargs_plottrend)
			
	stat = 'quantiles'
	if kwargs_trends[stat]:
		stat_func_kwargs['q'] = kwargs_trends['quantiles_q'] # add quantiles to kwargs
		func = stat_to_func[stat][kwargs_trends['ignore_nans']]
		xwise,ywise = xwise_ywise_stat(dat,func,**stat_func_kwargs)
		stat_func_kwargs.pop('q') # And get rid of it in case later
		
		for entry in kwargs_plottrend.keys(): kwargs_plottrend[entry] = kwargs_trends[stat+'_'+entry]
		
		for i in range(len(kwargs_trends['quantiles_q'])):
			ax_top.plot(x,xwise[i],**kwargs_plottrend)
			if repeat_on_x:
				ax_top.plot(x+x_offset,xwise[i],**kwargs_plottrend)
			ax_right.plot(ywise[i],y,**kwargs_plottrend)
	stat = 'std'
	if kwargs_trends[stat]:
		func = stat_to_func[stat][kwargs_trends['ignore_nans']]
		xwise,ywise = xwise_ywise_stat(dat,func,**stat_func_kwargs)
		
		for entry in kwargs_plottrend.keys(): kwargs_plottrend[entry] = kwargs_trends[stat+'_'+entry]
		
		stat_base = 'median' if kwargs_trends['median'] and not kwargs_trends['mean'] else 'mean'
		func_base = stat_to_func[stat_base][kwargs_trends['ignore_nans']]
		xwise_base,ywise_base = xwise_ywise_stat(dat,func_base,**stat_func_kwargs)
		
		for pm in [1.,-1.]:
			ax_top.plot(x,xwise_base + pm*xwise,**kwargs_plottrend)
			if repeat_on_x:
				ax_top.plot(x+x_offset,xwise_base + pm*xwise,**kwargs_plottrend)
			ax_right.plot(ywise_base + pm*ywise,y,**kwargs_plottrend)
	
	
	## Text
	if kwargs_infotxt['txt'] is not None:
		txt_infotxt = kwargs_infotxt.pop('txt')
		ax_infotxt.text(0.5,0.5,txt_infotxt, horizontalalignment='center',verticalalignment='top', transform=ax_infotxt.transAxes,**kwargs_infotxt)
	
	### Makeup
	
	# other colorbar stuff
	from matplotlib.pyplot import cm,colorbar
	from matplotlib.colors import Normalize
	cb_norm = Normalize(vmin=kwargs_axside['vmin_right'],vmax=kwargs_axside['vmax_right'])
	mappable,mappable._A = cm.ScalarMappable(cmap=cmap_topright, norm=cb_norm),[]
	cbar_other = colorbar(mappable,cax=cax_other,orientation='horizontal',pad=0.15)
	cbar_other_twin = cax_other.twiny()
	cbar_other_twin.set_xlim(kwargs_axside['vmin_top'],kwargs_axside['vmax_top'])
	
	# botright text
	ax_infotxt.axis("off")
	
	# Aux crap
	xlab = labels['x']
	ylab = labels['y']
	clab = labels['c']
	fontsize_lab = labels['fontsize']
	if xlab is not None:
		ax_main.set_xlabel(xlab,fontsize=fontsize_lab)
		cbar_other.set_label(xlab,fontsize=fontsize_lab)
	if ylab is not None:
		ax_main.set_ylabel(ylab,fontsize=fontsize_lab)
		cbar_other_twin.set_xlabel(ylab,fontsize=fontsize_lab)
	if clab is not None:
		cb_main.set_label(clab,fontsize=fontsize_lab)
		ax_top.set_ylabel(clab,fontsize=fontsize_lab)
		ax_right.set_xlabel(clab,fontsize=fontsize_lab)
	
	fig.align_labels() 
	
	# ticklabelz
	if repeat_on_x:
		import matplotlib.ticker as plticker
	
		
		def xtick_format(t,pos=None):
			return '%g'%(t) if t <= x[-1] else '%g'%(t-x_offset)
		ax_main.xaxis.set_major_formatter(plticker.FuncFormatter(xtick_format))
	
	plt.setp(ax_top.get_xticklabels()  , visible=False)
	plt.setp(ax_right.get_yticklabels(), visible=False)
	
	for ax in [ax_top ,ax_main,cax_main,ax_right,ax_infotxt,cax_other,cbar_other_twin]:
		ax.tick_params(axis='both', labelsize=fontsize_ticks )
	
	
	
	### Save or Show
	import sys
	if kwargs_saveshow['save']:
		figname = kwargs_saveshow['name']
		if figname is None:
			figname = sys._getframe().f_code.co_name
		
		full_path = kwargs_saveshow['path'] + figname +\
		            kwargs_saveshow['name_extra'] + kwargs_saveshow['ext']
		
		trysave = True
		
		while trysave:
			try:
				plt.savefig( full_path )
				trysave = False
			except PermissionError:
				input(" > PermissionError: Denied for %s, please close file window and press enter to try again"%( full_path ) )
			

	if kwargs_saveshow['show']:
		plt.show()
	
	return {'ax_main':ax_main,'ax_top':ax_top,'ax_right':ax_right}
	
def mk_mock_data():
	import numpy as np
	xmax = 10.
	nx = 11
	ny = 10
	x = np.linspace(0.,xmax,nx)
	y = np.linspace(-20.,2.,ny)
	
	tmpx = np.linspace( 0,xmax*ny,nx*ny )
	tmpy = np.linspace( y[0],y[-1],nx*ny)
	#~ dat = np.sin( 2.*np.pi*(tmpx-3.)/(xmax/3.) ) + tmpy
	dat = np.sin( 2.*np.pi*(tmpx-tmpy)/(xmax/3.) ) + tmpy
	dat = dat.reshape(len(y),len(x))
	
	for i,j in [ (5,7),(4,2) ]:
		dat[i,j] = np.nan
		
	return x,y,dat
	
def main():
	
	### Example
	# Forge some data
	x,y,dat = mk_mock_data()
	
	kwargs_infotxt = { 'txt': "Example\nTxt" }#,'c':'r','fontsize':14}
	labels = {'x': "Test", 'y': "Testy", 'c': "Color"}#, 'fontsize':21}
	
	kwargs_highlight = {'top': [-20.],'right':[5.,10.],'c': 'r','ls':'--'}
	
	kwargs_saveshow = {}#'show':False}
		
	axdict = plot_progression(x,y,dat,labels=labels,kwargs_highlight=kwargs_highlight,kwargs_saveshow=kwargs_saveshow)#,kwargs_infotxt=kwargs_infotxt)
	
	#~ axdict['ax_main'].plot([0,1],[0,1])
	#~ import matplotlib.pyplot as plt
	#~ plt.show()
	#~ plot_progression(x,y,dat,labels=labels,kwargs_infotxt=kwargs_infotxt)
	
	
if __name__ == "__main__":
	main()
