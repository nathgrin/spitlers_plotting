import numpy as np
import matplotlib.pyplot as plt

def differenceplot(xarr,yarr,**kwargs):
    
    c= kwargs.get('c','k')
    marker = kwargs.get('marker','o')
    ls = kwargs.get('ls','-')
    
    diff = np.diff(yarr)
    
    ax = plt.gca()
    
    if kwargs.get('zeroed_spines',True):
      ax.spines.left.set_position('zero')
      ax.spines.right.set_color('none')
      ax.spines.bottom.set_position('zero')
      ax.spines.top.set_color('none')
      ax.xaxis.set_ticks_position('bottom')
      ax.yaxis.set_ticks_position('left')
    if kwargs.get('grid',True):
      ax.grid(True)
    
    print(np.meshgrid(np.zeros(diff.shape),diff))
    for i,d in enumerate(diff):
        plt.plot([xarr[i+1],xarr[i+1]],[0,d],marker='',c=c,ls=ls)
    plt.plot(xarr[1:],diff,marker=marker,c=c,ls='')
    
    if kwargs.get('xticklabels',None) is not None:
      ax.set_xticks(xarr)
      ax.set_xticklabels(kwargs.get('xticklabels',None))
    
    plt.savefig('fig/diffplot.png')

def main():
    print('hi')
    
    vast_bedrag = 120
    verbruik = [20,40, 80, vast_bedrag, 140, 160]
    yarr = np.cumsum([ v-vast_bedrag for v in verbruik ])
    xticklabels = ['Juli','Augustus','September','Oktober','November','December']
    xarr = range(len(xticklabels))
    
    
    differenceplot(xarr,yarr,xticklabels=xticklabels)
  
if __name__ == "__main__":
    main()
