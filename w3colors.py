class color(object):
	def __init__(self,**kwargs):
		self.__dict__.update(kwargs)
	
	def __str__(self):
		attr_str = ""
		for attr,val in self.__dict__.items():
			attr_str += str(attr)+"="+str(val)+"; "
		return "<color obj: %s>"%(attr_str)
	
	def rgb_hex_to_rgb_int(self,rgb_hex_in=None):
		if rgb_hex_in is None:
			rgb_hex_in = self.getval('rgb_hex')
		
		rgb_hex_in = rgb_hex_in.lstrip('#')
		out = tuple(int(rgb_hex_in[i:i+2], 16) for i in (0, 2, 4))
		return out
	
	
	def rgb_int_to_hsv(self, rgb_int_in=None):
		if rgb_int_in is None:
			rgb_int_in = self.getval('rgb_int')
		
		mx = max(rgb_int_in)
		mn = min(rgb_int_in)
		df = mx-mn
		if mx == mn:
			h = 0
		elif mx == r:
			h = (60 * ((g-b)/df) + 360) % 360
		elif mx == g:
			h = (60 * ((b-r)/df) + 120) % 360
		elif mx == b:
			h = (60 * ((r-g)/df) + 240) % 360
		if mx == 0:
			s = 0
		else:
			s = (df/mx)*100
		v = mx*100/255.
		
		out = (h, s, v)
		
		return out
		
	
	# THESE ARE STOLEN FROM colorsys
	# https://github.com/python/cpython/blob/3.9/Lib/colorsys.py
	def _v(self,m1, m2, hue):
		hue = hue % 1.0
		if hue < 1./6.:
			return m1 + (m2-m1)*hue*6.0
		if hue < 0.5:
			return m2
		if hue < 1./3.:
			return m1 + (m2-m1)*(2./3.-hue)*6.0
		return m1
	
		
	def rgb_int_to_hsl(self,rgb_int_in=None):
		''' hsl all values between zero and 1.
		funciton stolen, but changed to have 0<=rgb_int<=255 '''
		if rgb_int_in is None:
			rgb_int_in = self.getval('rgb_int')
			
		r,g,b = rgb_int_in
		maxc = max(r, g, b)
		minc = min(r, g, b)
		# XXX Can optimize (maxc+minc) and (maxc-minc)
		l = (minc+maxc)/510.0 #2.
		if minc == maxc:
			return 0.0, l, 0.0
		if l <= 0.5:
			s = (maxc-minc) / (maxc+minc)
		else:
			s = (maxc-minc) / (510.0-maxc-minc) # 510=2.
		rc = (maxc-r) / (maxc-minc)
		gc = (maxc-g) / (maxc-minc)
		bc = (maxc-b) / (maxc-minc)
		if r == maxc:
			h = bc-gc
		elif g == maxc:
			h = 2.0+rc-bc
		else:
			h = 4.0+gc-rc
		h = (h/6.0) % 1.0
		
		out = (h,s,l)
		return out
	
	def hsl_to_rgb_int(self,hsl_in):
		if hsl_in is None:
			hsl_in = self.getval('hsl')
		
		h,s,l = hsl_in
		if s == 0.0:
			return l, l, l
		if l <= 0.5:
			m2 = l * (1.0+s)
		else:
			m2 = l+s-(l*s)
		m1 = 2.0*l - m2
		
		out = tuple(int(n*255) for n in (self._v(m1, m2, h+1./3.), self._v(m1, m2, h), self._v(m1, m2, h-1./3.)))
		return out
		
		
		
	def getval(self,which):
		
		if hasattr(self,which):
			return getattr(self,which)
		
		if which == 'hsl':
			self.hsl = self.rgb_int_to_hsl()
			return self.hsl
		elif which == 'rgb_int':
			self.rgb_int = self.rgb_hex_to_rgb_int()
			return self.rgb_int
		elif which == 'rgb_hex':
			raise Exception("I dont have rgb_hex :(")
			
		raise Exception("I dont have attribute %s :("%(which))
		return None
	
	def distance_to(self,other_color):
		if not isinstance(other_color,color):
			raise TypeError("distance_to requires input to be colorobject!")
		try:
			hsl_other = other_color.getval('hsl')
		except:
			raise Exception("other color no have hsl value :(")
		try:
			hsl_self = self.getval('hsl')
		except:
			raise Exception("I no have hsl value :(")
		
		
		
		return biconical_distsqr_between_hslcolors(hsl_self,hsl_other)
		
			
			

w3colors = []
def load_w3colors_file():
	
	w3colors.clear() # Empty list
	fname = "w3colors.txt"
	with open(fname) as thefile:
		for line in thefile:
			line = line.strip().split()
			if line[0] == "#": continue
			name = line[0]
			rgb_hex = line[1]
			rgb_int = tuple(int(n) for n in  line[2].split(','))
			
			w3colors.append( color(name=name,rgb_hex=rgb_hex,rgb_int=rgb_int) )
			
	return w3colors
	
def eucl_distsqr_between_rgbcolors(rgb_int_1,rgb_int_2):
	'''
	simple distance in euclidean rgb space (probably bad! :D)
	probably better to put em on a colorweel
	'''
	a = rgb_int_1[0]-rgb_int_2[0]
	b = rgb_int_1[1]-rgb_int_2[1]
	c = rgb_int_1[2]-rgb_int_2[2]
	distsqr = a*a+b*b+c*c
	return distsqr
def cil_distsqr_between_hslcolors(hsl_1,hsl_2,include_lightness=True):
	'''
	simple distance in euclidean cilindrical hsl space 
	i.e. cilindrical coordinates with h=phi s = r l=z
	'''
	from numpy import cos,pi
	
	a = hsl_1[1]*hsl_1[1]+hsl_2[1]*hsl_2[1]
	b = hsl_1[1]*hsl_2[1]*cos( 2.*pi*(hsl_1[0]-hsl_2[0]) )
	c = hsl_1[2]-hsl_2[2] if include_lightness else 0.
	distsqr = a-2.*b+c*c
	return distsqr
	
	
def maxchroma_hsl(lightness):
	'''
	see wiki
	'''
	return 1.-abs(2*lightness-1)
	
def biconical_distsqr_between_hslcolors(hsl_1,hsl_2,include_lightness=True):
	'''
	distance in euclidean biconical hsl space 
	i.e. cilindrical coordinates with h=phi s*maxchroma_hsl = r l=z
	see e.g., https://en.wikipedia.org/wiki/HSL_and_HSV
	'''
	from numpy import cos,pi
	
	r_1 = hsl_1[1]*maxchroma_hsl( hsl_1[2] )
	r_2 = hsl_2[1]*maxchroma_hsl( hsl_2[2] )
	
	a = r_1*r_1+r_2*r_2
	b = r_1*r_2*cos( 2.*pi*(hsl_1[0]-hsl_2[0]) )
	c = hsl_1[2]-hsl_2[2] if include_lightness else 0.
	distsqr = a-2.*b+c*c
	return distsqr
	

def rgb_hex_find_closest_named_color(rgb_hex_in):
	
	thecolor = color( rgb_hex = rgb_hex_in )
	thecolor.getval('rgb_int')
	thecolor.getval('hsl')
	
	# thecolor.rgb_int is generated when asked for first distance
	
	#~ mindist_color = min(w3colors, key=lambda other_color: thecolor.distance_to(other_color))
	
	print(" > Closest color to",thecolor,"is:")
	mindist_color = min(w3colors, key=lambda other_color: cil_distsqr_between_hslcolors(thecolor.getval('hsl'),other_color.getval('hsl'),include_lightness=True))
	print(" HSL CwL",mindist_color)
	
	mindist_color = min(w3colors, key=lambda other_color: cil_distsqr_between_hslcolors(thecolor.getval('hsl'),other_color.getval('hsl'),include_lightness=False))
	print(" HSL CnL",mindist_color)
	
	mindist_color = min(w3colors, key=lambda other_color: biconical_distsqr_between_hslcolors(thecolor.getval('hsl'),other_color.getval('hsl'),include_lightness=True))
	print(" HSL BwL",mindist_color)
	
	mindist_color = min(w3colors, key=lambda other_color: biconical_distsqr_between_hslcolors(thecolor.getval('hsl'),other_color.getval('hsl'),include_lightness=False))
	print(" HSL BnL",mindist_color)
	
	mindist_color = min(w3colors, key=lambda other_color: eucl_distsqr_between_rgbcolors(thecolor.getval('rgb_int'),other_color.getval('rgb_int')))
	print(" RGB    ",mindist_color)
	
	

def main():
	
	w3colors = load_w3colors_file()
	
	#~ cc = w3colors[1]
	#~ cc2 = w3colors[2]
	#~ print(cc,cc2)
	
	#~ cc.distance_to(cc2)
	#~ input()
	
	
	
	
	
	
	while True:
		print("> Input a hex of a color")
		userinput = input("?")
		rgb_hex_find_closest_named_color(userinput)
			
	
if __name__ == "__main__":
	main()
