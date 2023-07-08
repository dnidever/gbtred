import os
import errno
import numpy as np
from astropy.io import fits
from astropy.table import Table
from dlnpyutils import utils as dln,robust,plotting as pl
import matplotlib
import matplotlib.pyplot as plt

LIGHT_SPEED = 2.99792458e8   # speed of light in m/s

def getfreq(header):
    #crval1 = header['crval1']
    #cdelt1 = header['cdelt1']
    #naxis1 = header['naxis1']
    #crpix1 = header['crpix1']
    #freq = (np.arange(naxis1) + 1 - crpix1) * cdelt1 + crval1

    # from gbtidl/pro/toolbox/io/io_sdfits_line__define.pro
    # spec.reference_channel = self->get_row_value(row,'CRPIX1',virtuals,names,1.0) - 1.0
    reference_channel = header['crpix1'] - 1.0
    # spec.frequency_interval = self->get_row_value(row,'CDELT1',virtuals,names,dd)
    frequency_interval = header['cdelt1']
    # spec.reference_frequency = self->get_row_value(row,'CRVAL1',virtuals,names,dd)
    reference_frequency = header['crval1']
    if 'naxis1' in header:
        npix = header['naxis1']
    else:
        npix = len(header['data'].squeeze())
    
    # from gbtidl/pro/toolbox/chantofreq.pro    
    # start by constructing the frequency axis
    result = np.arange(npix)
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

def getvel(tab):
    """ Get velocity arrray."""
    
    # from gbtidl/pro/toolbox/io/io_sdfits_line__define.pro
    # spec.line_rest_frequency = self->get_row_value(row,'RESTFREQ',virtuals,names,dd)
    freq = getfreq(tab)
    vel = freqtovel(freq,tab['restfreq'])
    return vel
    
def dosingle(tab,binsize=20,npoly=5,verbose=True):
    """
    Remove baseline for a single spectrum
    """
    
    spec = tab['data'].data.data.squeeze()
    npix = len(spec)
    x = np.arange(npix)/(npix-1)  # scale from -1 to 1
    x = (x-0.5)*2
    vel = getvel(tab)
    
    # Bin the data
    smspec = dln.rebin(spec,binsize=binsize)
    smx = dln.rebin(x,binsize=binsize)
    smvel = dln.rebin(vel,binsize=binsize)
    nsmpix = len(smspec)
    
    # Fit polynomial and perform outlier rejection
    flag = True
    count = 0
    goodmask = (np.abs(smvel) > 30e3) & np.isfinite(smspec)  # always mask zero-velocity region
    last_sig = 999999.
    last_nmask = nsmpix-np.sum(goodmask)
    while (flag):
        if count == 0:
            npoly1 = np.minimum(npoly,3)
        elif count == 1:
            npoly1 = np.minimum(npoly,4) 
        else:
            npoly1 = npoly
        coef = robust.polyfit(smx[goodmask],smspec[goodmask],npoly1)
        model = np.polyval(coef,smx)
        resid = smspec-model
        med = np.nanmedian(resid)
        sig = dln.mad(resid)
        goodmask = ((np.abs(smvel) > 30e3) & (np.abs(resid) < 5*sig))
        nmask = nsmpix-np.sum(goodmask)
        if npoly1==npoly:
            if (count > 10) or ((last_sig-sig)/last_sig*100 < 5) or (nmask==last_nmask): flag=False
        last_sig = sig
        last_nmask = nmask
        count += 1
        if verbose:
            print(count,med,sig,nmask)
            print('  ',coef)
        
    model = np.polyval(coef,x)
    rspec = spec-model
        
    return rspec,model,coef


def dointegration(tab,npoly=5,verbose=True):
    """
    Baseline correct all spectra for a single integration.
    """

    nspec = len(tab)
    npix = len(tab[0]['data'])

    # Get full frequency range/array
    #  for both frequency positions
    frequency_interval = np.median(tab['cdelt1'])
    restfreq = np.median(tab['restfreq'])
    minfreq = 1e20
    maxfreq = -1e20
    for i in range(nspec):
        f = getfreq(tab[i])
        minfreq = np.min([np.min(f),minfreq])
        maxfreq = np.max([np.max(f),maxfreq])
    nallpix = int(np.round( (maxfreq-minfreq)/np.abs(frequency_interval) )) + 1
    header = {'crval1':maxfreq,'crpix1':1,'cdelt1':frequency_interval,'naxis1':nallpix,'restfreq':restfreq}
    allfreq = getfreq(header)
    allvel = getvel(header)
    
    # While loop until convergence
    flag = True
    count = 0
    combspec = None
    last_combspec = np.zeros(nallpix)+1e20
    while (flag):
        
        # Loop over the various spectra and do baseline correction
        specarr = np.zeros([nspec,nallpix],float)+np.nan
        coefarr = np.zeros([nspec,npoly+1],float)+np.nan
        for i in range(nspec):
            tab1 = tab[i:i+1].copy()
            freq = getfreq(tab1)
            # Get frequency range for this spectrum
            lo, = np.where(np.abs(allfreq-freq[0]) < 10)
            lo = lo[0]
            hi, = np.where(np.abs(allfreq-freq[-1]) < 10)
            hi = hi[0]
            # Baseline correction
            if combspec is None:
                rspec,model,coef = dosingle(tab1,verbose=verbose)                
            else:
            # Remove combined spectrum
                temp = tab1.copy()
                temp['data'] -= combspec[lo:hi+1]
                resid,model,coef = dosingle(temp,verbose=verbose)
                rspec = tab1['data'].data.data.squeeze() - model
            # Add to array
            specarr[i,lo:hi+1] = rspec
            coefarr[i,:] = coef
            
        # Combined spectrum
        ngood = np.sum(np.isfinite(specarr),axis=0)
        sumspec = np.nansum(specarr,axis=0)
        combspec = sumspec/ngood
        
        maxdiff = np.max(np.abs(combspec-last_combspec)) 
        if maxdiff < 0.05 or count>5: flag=False
        
        count += 1
        last_combspec = combspec
        if verbose:
            print(count,maxdiff)
        
    out = {'spec':combspec,'nspec':ngood,'freq':allfreq,'vel':allvel,'header':header,'coef':coefarr}
        
    return out



def session(filename,tag='_red',outfile=None,verbose=False):
    """
    Baseline correct a full session of data for a target/map.
    """

    # Load data
    if os.path.exists(filename)==False:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename)
    print('Processing ',filename)
    tab = Table.read(filename)
    for c in tab.colnames: tab[c].name = c.lower()

    if outfile is None:
        outfile = filename.replace('.fits',tag+'.fits')
    
    scans = np.unique(tab['scan'].data)
    print(len(scans),' scans')

    # Coordinate names
    lontype = tab['ctype2'][0].lower()
    lattype = tab['ctype3'][0].lower()
    
    # Scan loop
    finaldata = []
    for s in scans:
        print('=== Scan '+str(s)+' ===')
        sind, = np.where(tab['scan']==s)
        integrations = np.unique(tab['int'][sind].data)
        print(str(len(integrations))+' integrations')
        # Integration loop:
        for integ in integrations:
            iind, = np.where(tab['int'][sind]==integ)
            tab1 = tab[sind][iind]
            print('  Int '+str(integ)+'  '+str(len(tab1))+' spectra')
            # Make sure there is good ata
            if np.sum(np.isfinite(tab1['data'].data.data))==0:
                print('  No good data for this scan')
                continue
            sp = dointegration(tab1,verbose=False)
            sp['scan'] = s
            sp['int'] = integ
            sp[lontype] = tab1['crval2'][0]
            sp[lattype] = tab1['crval3'][0]
            finaldata.append(sp)
            
    # Reformat into a large table
    npix = len(finaldata[0]['spec'])
    dt = [('scan',int),('int',int),('data',float,npix),('nspec',int,npix),(lontype,float),(lattype,float)]
    final = np.zeros(len(finaldata),dtype=np.dtype(dt))
    for i in range(len(finaldata)):
        final['scan'][i] = finaldata[i]['scan']
        final['int'][i] = finaldata[i]['int']
        final['data'][i] = finaldata[i]['spec']
        final['nspec'][i] = finaldata[i]['nspec']  # how about exptime?
        final[lontype][i] = finaldata[i][lontype]
        final[lattype][i] = finaldata[i][lattype]        
        
    # Write the data out to a file
    # put velocity information in the header
    hdulist = fits.HDUList()
    hdulist.append(fits.table_to_hdu(Table(final)))
    vel = finaldata[0]['vel']
    hdulist.append(fits.ImageHDU(vel))
    hdulist[2].header['CRVAL1'] = vel[0]
    hdulist[2].header['CDELT1'] = vel[1]-vel[0]
    hdulist[2].header['CRPIX1'] = 1
    hdulist[2].header['NAXIS1'] = npix
    hdulist[2].header['CTYPE1'] = 'velocity'
    print('Writing data to ',outfile)
    hdulist.writeto(outfile,overwrite=True)
    
    #import pdb; pdb.set_trace()

    #return final,vel
