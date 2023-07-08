import os
import errno
import numpy as np
from astropy.io import fits
from astropy.table import Table

LIGHT_SPEED = 2.99792458e8   # speed of light in m/s

def getfreq(header):
    crval1 = header['crval1']
    cdelt1 = header['cdelt1']
    naxis1 = header['naxis1']
    crpix1 = header['crpix1']
    #freq = (np.arange(naxis1) + 1 - crpix1) * cdelt1 + crval1

    # from gbtidl/pro/toolbox/io/io_sdfits_line__define.pro
    # spec.reference_channel = self->get_row_value(row,'CRPIX1',virtuals,names,1.0) - 1.0
    reference_channel = header['CRPIX1'] - 1.0
    # spec.frequency_interval = self->get_row_value(row,'CDELT1',virtuals,names,dd)
    frequency_interval = header['CDELT1']
    # spec.reference_frequency = self->get_row_value(row,'CRVAL1',virtuals,names,dd)
    reference_frequency = header['CRVAL!']
    
    
    # from gbtidl/pro/toolbox/chantofreq.pro    
    # start by constructing the frequency axis
    result = np.arange(naxis1)
    result = result - reference_channel
    result = result * frequency_interval
    result = result + reference_frequency
    #offset = 0.0

    #result = freqtofreq(data, result, frame, data.frequency_type)

    return result

def freqtovel(freq, restfreq, veldef='RADIO'):
    """ Convert from frequency units to velocity units."""
    # from gbtidl/pro/toolbox/freqtovel.pro

    if veldef=='RADIO':
        result = LIGHT_SPEED * (1 - freq / restfreq)
    elif veldef=='OPTICAL':
        result = LIGHT_SPEED * (restfreq / freq - 1)
    elif veldef=='TRUE':
        g = (freq / restfreq)**2
        result = LIGHT_SPEED * (1 - g) / (1 + g) 
    else:
        raise ValueError('unrecognized velocity definition')

    return result

def dosingle(tab):
    """
    Remove baseline for a single spectrum
    """

    pass

def dointegration(tab):
    """
    Baseline correct all spectra for a single integration.
    """

    nspec = len(tab)
    
    # While loop until convergence
    flag = True
    count = 0
    combspec = None
    while (flag):
    
        # Loop over the various spectra and do baseline correction
        specarr = np.zeros([nspec,npix],float)+np.nan
        for i in range(nspec):
            spec = tab['data'][i]
            freq = np.arange(tab['naxis1'][i])*
            # Remove combined spectrum
            if combspec is not None:
                lo, = np.where(np.abs(combfreq-freq[0]) < 0.01)
                hi, = np.where(np.abs(combfreq-freq[-1]) < 0.01)
                spec -= combspec[lo:hi]
            # Baseline correction
            rspec,pars = dosingle(spec)
            # Add to array
            specarr[i,lo:hi] = rspec
            
        # Combined spectrum
        ngood = np.sum(np.isfinite(specarr),axis=0)
        sumspec = np.sum(specarr,axis=0)
        combspec = sumspec/ngood
        
    return combspec,rtab



def session(filename):
    """
    Baseline correct a full session of data for a target/map.
    """

    # Load data
    if os.path.exists(filename)==False:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename)
    tab = Table.read(filename)
    for c in tab.colnames: tab[c].name = c.lower()

    scans = np.unique(tab['scan'].data)
    
    # Scan loop
    for s in scans:
        sind, = np.where(tab['scan']==s)
        integrations = np.unique(tab['int'][sind].data)
        # Integration loop:
        for integ in integrations:
            iind, = np.where(tab['int'][sind]==integ)
            tab1 = tab[sind][iind]
            import pdb; pdb.set_trace()
            sp = dointegration(tab1)
