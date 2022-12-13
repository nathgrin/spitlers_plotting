
#~ import sys
#~ sys.path.insert(0,'../Python/')
# from misc_func import *
import numpy as np

"""
Adaptation of corner.py (https://corner.readthedocs.io/en/latest/)
"""

def main():
	
	from matplotlib.pyplot import show
	
	# Let's make some tests:
	from numpy.random import normal
	labels = [ str(x) for x in range(5) ]
	x,s,l= 0.,1.,100
	
	dataset_label = labels[0]
	
	bins = [10,10]
	
	datapoints_kwargs = {'ms':5,'marker':'^'}
	
	r1,r2,r3=normal(x,s,l),normal(x,s,l),normal(x,s,l)
	dd = mk_cornerplotlist(r1,r2,r3)
	fig = Ncorner(dd,datapoints_kwargs=datapoints_kwargs)#,dataset_label=dataset_label,bins=bins)
	#~ show()
	
	for i in range(1):##len(('k','r','b','darkorange','steelblue','hotpink','gold','maroon','darkgreen'))+1):
		dataset_label = labels[1]
		r1,r2,r3=normal(x,s,l),normal(x,s,l),normal(x,s,l)
		dd = mk_cornerplotlist(r1,r2,r3)
		Ncorner(dd,fig=fig,dataset_label=dataset_label)
	
	show()
	
	dataset_label = labels[2]
	r1,r2,r3=normal(x,s,l),normal(x,s,l),normal(x,s,l)
	dd = mk_cornerplotlist(r1,r2)#,r3)
	Ncorner(dd,fig=fig,dataset_label=dataset_label)
	
	show()
	
	r1,r2,r3=normal(x,s,l),normal(x,s,l),normal(x,s,l)
	dd = mk_cornerplotlist(r1,r2)#,r3)
	Ncorner(dd,fig=fig)
	
	
def mk_cornerplotlist(*args):
	return np.array(args).transpose()
	

def Ncorner(xs, bins=None, color=None, fig=None,
		   data_range=None, weights=None, 
		   smooth=None, smooth1d=None,
		   labels=None, label_kwargs=None,dataset_label=None,
		   show_titles='Npoints_binsize_nbins', title_fmt=".2f", title_kwargs=None,
		   truths=None, truth_color="#4682b4",
		   scale_hist=False, quantiles=None, verbose=True, 
		   max_n_ticks=5, top_ticks=False, use_math_text=False, reverse=False,share_axes=True,autoscale=True,axes_limits=None,
		   plot_datapoints=True,datapoints_kwargs=None,
		   plot_hist2d=False,plot_tracks=False,tracks_kwargs=None,
		   data_summary_print=True,data_summary_file='tmp_data_summary_corner.txt',data_summary_file_clear=True,
		   show_subplot_inds=False,
		   hist_kwargs=None, **hist2d_kwargs):
	"""
	stolen from corner module
	
	
	dataset_label = label of the dataset for legendi n topright corner.
	
	show_titles is a str that can contain:
	   - Npoints: show # of points
	   - nbins: show (# of bins)
	   - binsize: show (%.02f binsize)
	
	
	show_subplot_inds will put text( i,j [k]) with the indices and list entry of each plot.
	
	if color is None, decide it based on the mpl cycler  (if color is not already in the plot..)
	   Setting color can be dont with color= or in datapoints_kwargs.. the color= overrules whatever is written in datapoints_kwargs
	
	
	
	
	### From the docs:
	
	
	Make a *sick* corner plot showing the projections of a data set in a
	multi-dimensional space. kwargs are passed to hist2d() or used for
	`matplotlib` styling.

	Parameters
	----------
	xs : array_like[nsamples, ndim]
		The samples. This should be a 1- or 2-dimensional array. For a 1-D
		array this results in a simple histogram. For a 2-D array, the zeroth
		axis is the list of samples and the next axis are the dimensions of
		the space.

	bins : int or array_like[ndim,]
		The number of bins to use in histograms, either as a fixed value for
		all dimensions, or as a list of integers for each dimension.
		or 'Auto', freedman diaconis will be applied

	weights : array_like[nsamples,]
		The weight of each sample. If `None` (default), samples are given
		equal weight.

	color : str
		A ``matplotlib`` style color for all histograms.

	smooth, smooth1d : float
	   The standard deviation for Gaussian kernel passed to
	   `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms
	   respectively. If `None` (default), no smoothing is applied.

	labels : iterable (ndim,)
		A list of names for the dimensions. If a ``xs`` is a
		``pandas.DataFrame``, labels will default to column names.

	label_kwargs : dict
		Any extra keyword arguments to send to the `set_xlabel` and
		`set_ylabel` methods.

	show_titles : bool
		Displays a title above each 1-D histogram showing the 0.5 quantile
		with the upper and lower errors supplied by the quantiles argument.

	title_fmt : string
		The format string for the quantiles given in titles. If you explicitly
		set ``show_titles=True`` and ``title_fmt=None``, the labels will be
		shown as the titles. (default: ``.2f``)

	title_kwargs : dict
		Any extra keyword arguments to send to the `set_title` command.

	data_range : iterable (ndim,)
		A list where each element is either a length 2 tuple containing
		lower and upper bounds or a float in data_range (0., 1.)
		giving the fraction of samples to include in bounds, e.g.,
		[(0.,10.), (1.,5), 0.999, etc.].
		If a fraction, the bounds are chosen to be equal-tailed.

	truths : iterable (ndim,)
		A list of reference values to indicate on the plots.  Individual
		values can be omitted by using ``None``.

	truth_color : str
		A ``matplotlib`` style color for the ``truths`` makers.

	scale_hist : bool
		Should the 1-D histograms be scaled in such a way that the zero line
		is visible?

	quantiles : iterable
		A list of fractional quantiles to show on the 1-D histograms as
		vertical dashed lines.

	verbose : bool
		If true, print the values of the computed quantiles.

	plot_contours : bool
		Draw contours for dense regions of the plot.

	use_math_text : bool
		If true, then axis tick labels for very large or small exponents will
		be displayed as powers of 10 rather than using `e`.
		
	reverse : bool
		If true, plot the corner plot starting in the upper-right corner instead 
		of the usual bottom-left corner
		
	max_n_ticks: int
		Maximum number of ticks to try to use

	top_ticks : bool
		If true, label the top ticks of each axis

	fig : matplotlib.Figure
		Overplot onto the provided figure object.

	hist_kwargs : dict
		Any extra keyword arguments to send to the 1-D histogram plots.

	**hist2d_kwargs
		Any remaining keyword arguments are sent to `corner.hist2d` to generate
		the 2-D histogram plots.

	"""
	import logging
	import numpy as np
	import matplotlib.pyplot as pl
	from matplotlib.ticker import MaxNLocator, NullLocator
	from matplotlib.colors import LinearSegmentedColormap, colorConverter
	from matplotlib.ticker import ScalarFormatter
	
	if plot_tracks:
		trackdata = get_griddata()
		#~ print trackdata.keys()
	
	# Handle the kwargs
	if quantiles is None: quantiles = []
	if title_kwargs is None: title_kwargs = dict()
	if label_kwargs is None: label_kwargs = dict()
	if datapoints_kwargs is None: datapoints_kwargs = dict()
	if tracks_kwargs is None: tracks_kwargs = dict()
	if bins is None: bins = 'Auto'
	if hist_kwargs is None: hist_kwargs = dict()
	if dataset_label is None: dataset_label = ' '
	
	
	
	
	# Deal with 1D sample lists.
	#~ print xs
	#~ print xs.shape
	xs = np.atleast_1d(xs)
	if len(xs.shape) == 1:
		xs = np.atleast_2d(xs)
	else:
		assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
		xs = xs.T
	#~ print xs
	#~ print xs.shape
	#~ if xs.shape[0] <= xs.shape[1]:
		#~ print " == Warning, Ncorner: Your list contains more variables than datapoints!"
	
	# Try filling in labels from pandas.DataFrame columns.
	if labels is None:
		if fig is not None: # Try to get them from the figure
			labels = [fig.axes[len(xs)*(len(xs)-1)].xaxis.get_label().get_text()]
			for i in range(len(xs)-1): labels.append(fig.axes[len(xs)*(i+1)].yaxis.get_label().get_text())
		else:
			labels = ['' for x in range(len(xs)) ]
		#~ print labels
		#~ print labels
	if len(labels) != len(xs):
		raise ValueError("Ncorner: Your labels are not the same length as your data!!!")
			
	if autoscale and axes_limits is not None:
		print(" == Warning, Ncorner: Why do you have both autoscale and axes_limits? I dont know what to do..")
	if not autoscale and axes_limits is None:
		print(" == Warning, Ncorner: You dont want to autoscale and didnt give axes_limits\n you are being inconsistent and i am going to autoscale nonetheless")
		autoscale = True
	if axes_limits is not None:
		if len(axes_limits) != len(labels):
			raise ValueError("Ncorner: Your axes_limits do not have the same length as your data/labels..")
		
	if not all (len(x)==len(xs[0]) for x in xs):
		raise ValueError("Ncorner: Your columns have all different lengths: %s, why do you do this? I need a rectangular matrix.."%(str([len(x)for x in xs])))
	
	#~ raw_input()
	
	
	### Setup the Data
	

	# Parse the weight array.
	if weights is not None:
		weights = np.asarray(weights)
		if weights.ndim != 1:
			raise ValueError("Weights must be 1-D")
		if xs.shape[1] != weights.shape[0]:
			raise ValueError("Lengths of weights must match number of samples")
	
	# Get rid of irregularities (inf)
	ind_inf = np.isinf(xs)
	if np.sum(ind_inf) > 0:
		print(" == Warning, Ncorner: you have %i infinities in your sample, I REPLACED THEM BY NaN"%(np.sum(ind_inf)) )
		xs[ind_inf] = float('NaN')
	

	# Parse the parameter data_ranges.
	if data_range is None:
		if "extents" in hist2d_kwargs:
			logging.warn("Deprecated keyword argument 'extents'. "
						 "Use 'data_range' instead.")
			data_range = hist2d_kwargs.pop("extents")
		else:
			#~ data_range = [[x.min(), x.max()] for x in xs]
			data_range = [[np.nanmin(x), np.nanmax(x)] for x in xs]
			for i in range(len(data_range)):
				if data_range[i][0] != data_range[i][0]: data_range[i][0] = 0.
				if data_range[i][1] != data_range[i][1]: data_range[i][1] = 0.1
				
			#~ print data_range
			# Check for parameters that never change.
			m = np.array([e[0] == e[1] for e in data_range], dtype=bool)
			#~ if np.any(m):
				#~ raise ValueError(("It looks like the parameter(s) in "
								  #~ "column(s) {0} have no dynamic data_range. "
								  #~ "Please provide a `data_range` argument.")
								 #~ .format(", ".join(map(
									 #~ "{0}".format, np.adata_range(len(m))[m]))))

	else:
		# If any of the extents are percentiles, convert them to data_ranges.
		# Also make sure it's a normal list.
		data_range = list(data_range)
		for i, _ in enumerate(data_range):
			try:
				emin, emax = data_range[i]
			except TypeError:
				raise ValueError("I have no clue what im doing here")
				q = [0.5 - 0.5*data_range[i], 0.5 + 0.5*data_range[i]]
				data_range[i] = quantile(xs[i], q, weights=weights)
	
	
	if len(data_range) != xs.shape[0]:
		raise ValueError("Dimension mismatch between samples and data_range")
	
	# Parse the bin specifications.
	if type(bins) == int:
		bins = [int(bins) for _ in xs]
	elif type(bins) == list:
		if not all (type(b)==int for b in bins):
			raise ValueError('Ncorner: if bins is list, please provide a list of ints')
		if len(bins) != len(xs):
			raise ValueError("Ncorner: Please provide a list of ints as long as the data..")
	elif type(bins) ==str:
		if bins == 'Auto':
			bins = [apply_freedman_diaconis(xxx) for xxx in xs]
		else:
			raise ValueError('Ncorner: if bins is str, please say Auto')
	else:
		raise ValueError("Ncorner: Please give bins as single int, list of int or str='Auto'")
	
	
	### Setup the Figure
	
	# Some magic numbers for pretty axis layout.
	K = len(xs)
	factor = 2.0		   # size of one side of one panel
	if reverse:
		lbdim = 0.2 * factor   # size of left/bottom margin
		trdim = 0.5 * factor   # size of top/right margin
	else:
		lbdim = 0.5 * factor   # size of left/bottom margin
		trdim = 0.2 * factor   # size of top/right margin
	whspace = 0.05		 # w/hspace size
	plotdim = factor * K + factor * (K - 1.) * whspace
	dim = lbdim + plotdim + trdim

	# Create a new figure if one wasn't provided.
	if fig is None:
		fig, axarr = pl.subplots(K, K, figsize=(dim, dim))
		
	else:
		try:
			axarr = np.array(fig.axes).reshape((K, K))
			data_summary_file_clear = False # dont clear the summaryfile
		except:
			raise ValueError("Provided figure has {0} axes, but data has "
							 "dimensions K={1}".format(len(fig.axes), K))
	
	# Format the figure.
	lb = lbdim / dim
	tr = (lbdim + plotdim) / dim
	fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
						wspace=whspace, hspace=whspace)
	
	
	### Setup some more
	
	# Default coloring
	if color is None:
		color = datapoints_kwargs.get('c',None)
		if color is None:
			color = datapoints_kwargs.get('color',None)
			if color is None:
				ax_getlines = axarr[1,0]._get_lines
				c_list = [ l.get_color() for l in axarr[1,0].lines ]
				color = ax_getlines.get_next_color()
				if color in c_list:
					first_c = color
					while True:
						color = ax_getlines.get_next_color()
						if color is first_c: break
						if color not in c_list: break
	
	
	
	# Set up the default histogram keywords. # this should belong with the other histogram stuff upstairs.
	hist_kwargs["color"] = hist_kwargs.get("color", color)
	if smooth1d is None:
		hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")
	
	
	
	
	
	def shortcut_axloop():
		pass
		
	
	for i, x in enumerate(xs):
		
		# Link the axes
		if share_axes:
			xshares = [ axarr[i+j,i] for j in range(K-i) ]
			yshares = [ axarr[i,j] for j in range(i+int(i==K-1)) ]
			set_share_axes( xshares,sharex=True )
			set_share_axes( yshares,sharey=True )
			
			## This was an attempt to share x with y axes, but doesnt work
			#~ if i>0 and i<K-1:
				#~ print i
				#~ def on_y_change(cur_ax):
					#~ axarr[i,i].set_xlim(axarr[0,i].get_ylim())
				#~ axarr[0,i].callbacks.connect('ylim_changed',on_y_change)
		
		# Deal with masked arrays.
		if hasattr(x, "compressed"):
			x = x.compressed()

		if np.shape(xs)[0] == 1:
			ax = axarr
		else:
			if reverse:
				ax = axarr[K-i-1, K-i-1]
			else:
				ax = axarr[i, i]
		# Plot the histograms.
		if smooth1d is None:
			orientation = "horizontal" if i == K-1 and share_axes else "vertical"
			#~ print np.sort(data_range[i])
			n, _, _ = ax.hist(x[~np.isnan(x)], bins=bins[i], weights=weights,
							  range=np.sort(data_range[i]), orientation=orientation, **hist_kwargs )
		else:
			if gaussian_filter is None:
				raise ImportError("Please install scipy for smoothing")
			n, b = np.histogram(x, bins=bins[i], weights=weights,
								data_range=np.sort(data_range[i]))
			n = gaussian_filter(n, smooth1d)
			x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
			y0 = np.array(list(zip(n, n))).flatten()
			
			ax.plot(y0, x0, **hist_kwargs) if i == K-1 and share_axes else ax.plot(x0, y0, **hist_kwargs)
		
		# plot zeroline for hist
		ax.axvline(0,c='k',lw=0.5) if i == K-1 and share_axes else ax.axhline(0,c='k',lw=0.5)

		if truths is not None and truths[i] is not None:
			ax.axvline(truths[i], color=truth_color)

		# Plot quantiles if wanted.
		if len(quantiles) > 0:
			qvalues = quantile(x, quantiles, weights=weights)
			for q in qvalues:
				
				ax.axhline(q, ls="dashed", color=color) if i == K-1 and share_axes else ax.axvline(q, ls="dashed", color=color)

			if verbose:
				print("%i, %s, Quantiles:"%(i,labels[i]))
				print([item for item in zip(quantiles, qvalues)])

		if show_titles:
			title = None
			if title_fmt is not None:
				if False: # quantile stuff..
					# Compute the quantiles for the title. This might redo
					# unneeded computation but who cares.
					q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84],
												weights=weights)
					q_m, q_p = q_50-q_16, q_84-q_50
	
					# Format the quantile display.
					fmt = "{{0:{0}}}".format(title_fmt).format
					title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
					title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
	
					# Add in the column name if it's given.
					if labels is not None:
						title = "{0} = {1}".format(labels[i], title)
				
				
				# Set the title to # of non nans
				if type(show_titles)==str:
					title = ""
					if 'Npoints' in show_titles:
						title = str(sum(~np.isnan(x)))
					if 'nbins' in show_titles and 'binsize' in show_titles:
						title += ' (%i, %.2f)'%(bins[i],abs(data_range[i][1]-data_range[i][0])/bins[i])
					elif 'nbins' in show_titles:
						title += ' (%i)'%(bins[i])
					elif 'binsize' in show_titles:
						title += ' (%.2f)'%(abs(data_range[i][1]-data_range[i][0])/bins[i])
				
			elif labels is not None:
				title = "{0}".format(labels[i])
			

			if title is not None:
				if reverse:
					ax.set_xlabel(title, **title_kwargs)
				else:
					ax.set_title(title,fontsize=16, **title_kwargs)

		# Set up the axes.
		ax.set_xlim(data_range[i])
		if scale_hist:
			maxn = np.max(n)
			ax.set_xlim(-0.1 * maxn, 1.1 * maxn) if i == K-1 and share_axes else ax.set_ylim(-0.1 * maxn, 1.1 * maxn) 
		else:
			
			ax.set_xlim(0, 1.1 * np.max(n)) if i == K-1 and share_axes else ax.set_ylim(0, 1.1 * np.max(n))
				
		
		ax.set_xticklabels([]) 
		ax.yaxis.tick_right() if i < K-1 else ax.set_yticklabels([])
		[l.set_fontsize(10) for l in ax.get_yticklabels()] if i < K-1 else [l.set_fontsize(10) for l in ax.get_xticklabels()]
		
		if max_n_ticks == 0:
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
		else:
			ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
			#~ ax.yaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
			#~ ax.yaxis.set_ticks_position("right")
			#~ ax.xaxis.set_ticks_position("top")

		if i < K - 1:
			if top_ticks:
				ax.xaxis.set_ticks_position("top")
				#~ [l.set_rotation(45) for l in ax.get_xticklabels()]
			#~ else:
				#~ ax.set_xticklabels([])
		else:
			if reverse:
				ax.xaxis.tick_top()
			#~ [l.set_rotation(45) for l in ax.get_xticklabels()]
			#~ if labels is not None:
				#~ if reverse:
					#~ ax.set_title(labels[i], y=1.25, **label_kwargs)
				#~ else:
					#~ print i,labels[i]
					#~ ax.set_xlabel(labels[i], **label_kwargs)

			# use MathText for axes ticks
			ax.xaxis.set_major_formatter(
				ScalarFormatter(useMathText=use_math_text))


		# Now do the plots in the corner
		for j, y in enumerate(xs):
			
			if np.shape(xs)[0] == 1:
				ax = axarr
			else:
				if reverse:
					ax = axarr[K-i-1, K-j-1]
				else:
					ax = axarr[i, j]
					
					
			if show_subplot_inds: # Show the indices of the subplots
				
				ax.text(0.5,0.5,'[%i,%i [%i]]'%(i,j,fig.axes.index(ax)),horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,zorder=99)
				
			
					
			if j > i:
				ax.set_frame_on(False)
				ax.set_xticks([])
				ax.set_yticks([])
				continue
			elif j == i: # histograms already done above
				continue

			# Deal with masked arrays.
			if hasattr(y, "compressed"):
				y = y.compressed()
			
			if plot_hist2d:
				#~ raw_input('plothist2d')
				Nhist2d(y, x, ax=ax, range=[data_range[j], data_range[i]], weights=weights,
				  color=color, smooth=smooth, bins=[bins[j], bins[i]],
				  **hist2d_kwargs)
			
			# PLOT THE DATAPOINTS
			def shortcut_plotdata():
				pass
			if plot_datapoints:
				# IF you change something here YOU HAVE TO
				#    change it downstairs as well if you want
				#    the legend to go with it
				datapoints_alpha = datapoints_kwargs.get('alpha',1.)
				datapoints_ms = datapoints_kwargs.get('ms',2.)
				datapoints_marker = datapoints_kwargs.get('marker','.')
				datapoints_zorder = datapoints_kwargs.get('zorder',1.)
				datapoints_mew = datapoints_kwargs.get('mew',1.)
				datapoints_mfc = datapoints_kwargs.get('mfc',None)
				
				ax.plot(y,x,marker=datapoints_marker,ls='',c=color,ms=datapoints_ms,zorder=datapoints_zorder,alpha=datapoints_alpha,mew=datapoints_mew,mfc=datapoints_mfc)
			
			if plot_tracks:
				colors = ['r','g','b','maroon','gold']
				masses = ['008.00','015.00','020.00','025.00','040.00']
				
				tracks_lw = tracks_kwargs.get('lw',2)
				tracks_zorder = tracks_kwargs.get('zorder',10)
				for kk,mass in enumerate(masses):#,'020.00','040.00']:
					xdata = get_plotdata_tracks(trackdata[mass],labels[j])
					ydata = get_plotdata_tracks(trackdata[mass],labels[i])
					#~ print i,j,labels[i],labels[j]
					#~ print ' ',xdata,ydata
					ax.plot(xdata,ydata,zorder=tracks_zorder,c=colors[kk],lw=tracks_lw)
			
			if not autoscale and axes_limits is not None:
				ax.set_xlim(axes_limits[j])
				ax.set_ylim(axes_limits[i])
				#~ raw_input()
				
			if truths is not None:
				if truths[i] is not None and truths[j] is not None:
					ax.plot(truths[j], truths[i], "s", color=truth_color)
				if truths[j] is not None:
					ax.axvline(truths[j], color=truth_color)
				if truths[i] is not None:
					ax.axhline(truths[i], color=truth_color)

			if max_n_ticks == 0:
				ax.xaxis.set_major_locator(NullLocator())
				ax.yaxis.set_major_locator(NullLocator())
			else:
				ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
													   prune="lower"))
				ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
													   prune="lower"))

			if i < K - 1:
				ax.set_xticklabels([])
			else:
				if reverse:
					ax.xaxis.tick_top()
				#~ [l.set_rotation(45) for l in ax.get_xticklabels()]
				if labels is not None:
					ax.set_xlabel(labels[j], **label_kwargs)
					if reverse:
						ax.xaxis.set_label_coords(0.5, 1.4)
					else:
						ax.xaxis.set_label_coords(0.5, -0.3)

				# use MathText for axes ticks
				ax.xaxis.set_major_formatter(
					ScalarFormatter(useMathText=use_math_text))

			if j > 0:
				ax.set_yticklabels([])
			else:
				if reverse:
					ax.yaxis.tick_right()
				#~ [l.set_rotation(45) for l in ax.get_yticklabels()]
				if labels is not None:
					if reverse:
						ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
						ax.yaxis.set_label_coords(1.3, 0.5)
					else:
						ax.set_ylabel(labels[i], **label_kwargs)
						ax.yaxis.set_label_coords(-0.3, 0.5)

				# use MathText for axes ticks
				ax.yaxis.set_major_formatter(
					ScalarFormatter(useMathText=use_math_text))
					
					
	### Some extra info for the stuff
	def shortcut_extras():
		return
	### Legend for data
	if not plot_datapoints:
		datapoints_marker = 'o'
		datapoints_mew = 1
	#~ print rc
	axarr[0,K-1].plot(0,0,c=color,marker=datapoints_marker,mew=datapoints_mew,ls='',mfc=datapoints_mfc,label=dataset_label,zorder=0)
	#~ axarr[0,K-1].legend(loc=(0.5,1.),borderpad=0,facecolor='w',fontsize=16,framealpha=1,handletextpad=0.15,handlelength=1,edgecolor='w')#,zorder=10)
	axarr[0,K-1].legend(loc=9,borderpad=0,facecolor='w',fontsize=16,framealpha=1,handletextpad=0.15,handlelength=1,edgecolor='w',bbox_to_anchor=(0.5,1.),borderaxespad=0.)#,labelspacing=.2)#,zorder=10)
	axarr[0,K-1].set_xlim(100,101)
	if len(dataset_label) > 18:
		print('   -- Warning, Ncorner: avoid len(dataset_label)>18 for cosmetic reasons')
	#~ print 'a',
	#~ raw_input()
	
	
	# number of points as text
	from matplotlib import text
	axes_child = axarr[0,K-2].get_children()[0]
	if type(axes_child) == text.Text:
		writestr = axes_child.get_text() + '\n'
	else:
		writestr = ""
		axes_child = axarr[0,K-2].text(1.,1., writestr , fontsize=16,horizontalalignment='right', verticalalignment='top', transform=axarr[0,K-2].transAxes,linespacing=1.35)#,backgroundcolor='r')
	writestr += str(len(xs[0]))
	#~ print writestr
	#~ raw_input()
	axes_child.set_text(writestr)
	
	
	if plot_tracks:
		from matplotlib.lines import Line2D
		for kk,mass in enumerate(masses): # defined somewhere above (i think in the loop..)
			custom_lines = [ Line2D([0],[0],color=colors[kk],lw=4) for kk,mass in enumerate(masses) ]
			
			axarr[1,K-1].legend(custom_lines,masses,fontsize=14)#loc=3,bbox_to_anchor=(0.5, 1.),
			#~ custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                #~ Line2D([0], [0], color=cmap(.5), lw=4),
                #~ Line2D([0], [0], color=cmap(1.), lw=4)]
		
	
				
	### autoscale in hindsight
	
	for i in range(K):
		if autoscale: # scale both axis
			axarr[i,i].autoscale()
		else:# scale the histograms y-axis anyway
			axarr[i,i].autoscale()
			axarr[i,i].set_xlim(axes_limits[i]) if i != K-1 else axarr[i,i].set_ylim(axes_limits[i])
			
					
					
	### Auxiliary
	def mk_data_summary():
		
		if data_summary_file:
			datasummary_file = manipulate_files(data_summary_file)
			if data_summary_file_clear:
				datasummary_file.clearfile()
			from datetime import datetime
			datasummary_file.write_appline('\n=== ===\n%s'%(str(datetime.now())))
		def mkrow(dolist):
			collength = 10
			writestr = ""
			for dostr in dolist: 
				#~ print  type(dostr)
				if type(dostr) != str:
					if type(dostr) == np.float64:
						#~ print dostr
						#~ dostr = "{:8.3f}".format(dostr)
						dostr = "%+08.04f"%(dostr)
						#~ print dostr
					else:
						dostr = str(dostr)
					
				writestr+= dostr.ljust(collength)
			return writestr
		writestr = '\n Data Summary for %s (tot %i):'%(dataset_label,len(xs[0]))
		writestr += '\n'+ mkrow(['col','~nan','mean','median','std','IQR','min','max'])
		if data_summary_print: print( writestr )
		if data_summary_file: datasummary_file.write_appline(writestr)
			
		for i in range(len(xs)):
			x = xs[i]
			thenotnans = ~np.isnan(x)
			rowdata = [labels[i],sum(thenotnans)]
			if len(x)==sum(~thenotnans):
				rowdata += [float('NaN') for n in range(4)]
			else:
				rowdata += [np.mean(x[thenotnans]),np.median(x[thenotnans]),np.std(x[thenotnans]),np.subtract(*np.percentile(x[thenotnans], [75, 25])),min(x[thenotnans]),max(x[thenotnans])]
			writestr = mkrow(rowdata)
			if data_summary_print: print( writestr )
			if data_summary_file: datasummary_file.write_appline(writestr)
		
		
	if data_summary_print or data_summary_file:
		mk_data_summary()
	

	return fig
	
def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    stolen form an answer on stackoverflow, thanks!
    (doesn't seem to work because .. probably something wrong with yd or whatever
    """

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        # print yd
        # print type(yd)
        # print ((xd>lo) & (xd<hi))
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if len(axs) <= 1: return
    if target is None:
        target = axs[0]
    # Manage share using grouper objects
    for ax in axs:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    #~ if sharex and axs.ndim > 1:
        #~ for ax in axs[:-1,:].flat:
            #~ ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            #~ ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    #~ if sharey and axs.ndim > 1:
        #~ for ax in axs[:,1:].flat:
            #~ ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            #~ ax.yaxis.offsetText.set_visible(False)

def Nhist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
		   ax=None, color=None, plot_datapoints=False, plot_density=False,
		   plot_contours=False, no_fill_contours=True, fill_contours=False,
		   contour_kwargs=None, contourf_kwargs=None, data_kwargs=None, plot_colorbar=False,
		   **kwargs):
	"""
	Plot a 2-D histogram of samples.

	Parameters
	----------
	x : array_like[nsamples,]
	   The samples.

	y : array_like[nsamples,]
	   The samples.

	levels : array_like
		The contour levels to draw.

	ax : matplotlib.Axes
		A axes instance on which to add the 2-D histogram.

	plot_datapoints : bool
		Draw the individual data points.

	plot_density : bool
		Draw the density colormap.

	plot_contours : bool
		Draw the contours.

	no_fill_contours : bool
		Add no filling at all to the contours (unlike setting
		``fill_contours=False``, which still adds a white fill at the densest
		points).

	fill_contours : bool
		Fill the contours.

	contour_kwargs : dict
		Any additional keyword arguments to pass to the `contour` method.

	contourf_kwargs : dict
		Any additional keyword arguments to pass to the `contourf` method.

	data_kwargs : dict
		Any additional keyword arguments to pass to the `plot` method when
		adding the individual data points.

	"""
	import logging
	import numpy as np
	import matplotlib.pyplot as pl
	from matplotlib.ticker import MaxNLocator, NullLocator
	from matplotlib.colors import LinearSegmentedColormap, colorConverter
	from matplotlib.ticker import ScalarFormatter
	
	if ax is None:
		ax = pl.gca()

	# Set the default range based on the data range if not provided.
	if range is None:
		if "extent" in kwargs:
			logging.warn("Deprecated keyword argument 'extent'. "
						 "Use 'range' instead.")
			range = kwargs["extent"]
		else:
			range = [[x.min(), x.max()], [y.min(), y.max()]]

	# This is the color map for the density plot, over-plotted to indicate the
	# density of the points near the center.
	density_cmap = LinearSegmentedColormap.from_list(
		"density_cmap", [(1, 1, 1, 0), color])
	

	# We'll make the 2D histogram to directly estimate the density.
	try:
		H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
								 range=list(map(np.sort, range)),
								 weights=weights)
	except ValueError:
		raise ValueError("It looks like at least one of your sample columns "
						 "have no dynamic range. You could try using the "
						 "'range' argument.")

	if smooth is not None:
		if gaussian_filter is None:
			raise ImportError("Please install scipy for smoothing")
		H = gaussian_filter(H, smooth)

	from matplotlib.colors import LogNorm
	norm = LogNorm(vmin=0.1)
	ax.pcolor(X, Y, H.T, cmap=density_cmap,norm=norm)
		
if __name__ == "__main__":
	main()
