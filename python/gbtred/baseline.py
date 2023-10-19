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
    try:
        if 'naxis1' in header:
            npix = header['naxis1']
        else:
            npix = len(header['data'].squeeze())
    except:
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
    """
    Convert from frequency units to velocity units.

    Parameters
    ----------
    freq : numpy array

    restfreq : float

    velfreq : str, optional

    Returns
    -------
    result : numpy array
       

    Example
    -------

    f = freqtovel(freq, restfreq)

    """
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
    
def dosingle(tab,binsize=200,smlen=10,npoly=5,maskvel=30e3,edgetrim=12000,velrange=None,verbose=True):
    """
    Remove baseline for a single spectrum
    """

    try:
        spec = tab['data'].data.data.squeeze()
    except:
        spec = tab['data'].data.squeeze()
    npix = len(spec)
    px = np.arange(npix)
    x = px/(npix-1)  # scale from -1 to 1
    x = (x-0.5)*2
    vel = getvel(tab)
    
    # Bin the data
    smspec = dln.rebin(spec,binsize=binsize)
    smx = dln.rebin(x,binsize=binsize)
    smpx = dln.rebin(px,binsize=binsize)
    smvel = dln.rebin(vel,binsize=binsize)
    nsmpix = len(smx)
    
    # Fit polynomial and perform outlier rejection
    flag = True
    count = 0
    goodmask = (np.abs(smvel) > maskvel) & np.isfinite(smspec)  # always mask zero-velocity region
    if velrange is not None:
        godmask = goodmask & (smvel>velrange[0]) & (smvel<velrange[1])
    # mask negative zero-velocity regions as well
    if edgetrim is not None:
        goodmask = goodmask & (smpx>edgetrim) & (smpx<(npix-edgetrim))
    last_sig = 999999.
    last_nmask = nsmpix-np.sum(goodmask)
    while (flag):
        if count == 0:
            npoly1 = np.minimum(npoly,3)
        elif count == 1:
            npoly1 = np.minimum(npoly,4) 
        else:
            npoly1 = npoly
        # Some Gaussian smoothing
        #temp = bspec.copy()
        #temp[~goodmask] = np.nan
        #smspec = dln.gsmooth(temp,smlen)
        # Fit polynomial
        coef = robust.polyfit(smx[goodmask],smspec[goodmask],npoly1)
        model = np.polyval(coef,smx)
        resid = smspec-model
        med = np.nanmedian(resid)
        sig = dln.mad(resid)
        goodmask = ((np.abs(smvel) > maskvel) & (np.abs(resid) < 5*sig))
        # grow the bad regions?
        if velrange is not None:
            godmask = goodmask & (smvel>velrange[0]) & (smvel<velrange[1])        
        if edgetrim is not None:
            goodmask = goodmask & (smpx>edgetrim) & (smpx<(npix-edgetrim))
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

    #import pdb; pdb.set_trace()
    
    return rspec,model,coef


def dointegration(tab,npoly=5,maskvel=30e3,edgetrim=12000,maxiter=5,velrange=None,verbose=True):
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
                rspec,model,coef = dosingle(tab1,maskvel=maskvel,
                                            edgetrim=edgetrim,verbose=verbose) 
            else:
            # Remove combined spectrum
                temp = tab1.copy()
                temp['data'] -= combspec[lo:hi+1]
                resid,model,coef = dosingle(temp,maskvel=maskvel,
                                            edgetrim=edgetrim,verbose=verbose)
                try:
                    rspec = tab1['data'].data.data.squeeze() - model
                except:
                    rspec = tab1['data'].data.squeeze() - model                    
            # Edgetrim                
            if edgetrim is not None:
                rspec[0:int(edgetrim)] = np.nan
                rspec[-int(edgetrim):] = np.nan        
            # Add to array
            specarr[i,lo:hi+1] = rspec
            coefarr[i,:] = coef
            
        # Combined spectrum
        ngood = np.sum(np.isfinite(specarr),axis=0)
        sumspec = np.nansum(specarr,axis=0)
        combspec = sumspec/ngood
        
        maxdiff = np.max(np.abs(combspec-last_combspec)) 
        if maxdiff < 0.05 or count>maxiter: flag=False
        
        count += 1
        last_combspec = combspec
        if verbose:
            print(count,maxdiff)

    # Calculate the switching amount
    crval1 = np.array([t['crval1'] for t in tab])
    freq_switch_offset = np.abs(np.max(crval1)-np.min(crval1))

    #import pdb; pdb.set_trace()
    
    ## Trim edges
    #if edgetrim is not None:
    #    lo = int(edgetrim)
    #    hi = int(npix-edgetrim)
    #    combspec = combspec[lo:hi]
    #    ngood = ngood[lo:hi]
    #    allfreq = allfreq[lo:hi]
    #    allvel = allvel[lo:hi]
    #    header['crpix1'] = header['crpix1']-edgetrim       # only change the reference pixel position
    #    header['naxis1'] = len(combspec)         
    #    # coefarr uses original x values (from -1 to +1)
    
    # Trim in velocity
    if velrange is not None:
        lo = np.where(allvel >= velrange[0])[0][0]
        hi = np.where(allvel <= velrange[1])[0][-1]
        combspec = combspec[lo:hi+1]
        ngood = ngood[lo:hi+1]
        allfreq = allfreq[lo:hi+1]
        allvel = allvel[lo:hi+1]
        header['crpix1'] = 1
        header['crval1'] = allfreq[0]        
        header['naxis1'] = len(combspec)         
        # coefarr uses original x values (from -1 to +1)
    
    out = {'spec':combspec,'nspec':ngood,'freq':allfreq,'vel':allvel,
           'freq_switch_offset':freq_switch_offset,
           'header':header,'coef':coefarr}

    # put in single dish format
    # average 4 spectrum values
    # 
    
    return out

def cleansidelobes(sp,smlen=5,verbose=False):
    """
    Clean up the sidelobes by iteratively subtracting the signal.
    """

    spec = sp['spec']
    vel = sp['vel'] / 1e3
    freq = sp['freq']
    npix = len(spec)
    
    freq_switch_offset_hz = sp['freq_switch_offset']  # in Hz
    freq_switch_offset_chan =  int( freq_switch_offset_hz / np.abs(sp['header']['cdelt1']) )

    # Iteration
    vmaxarr = [50,100,200,300]
    bestspec = spec
    for i in range(len(vmaxarr)):
        vmax = vmaxarr[i]
        if verbose:
            print(i+1,' vmax: ',vmax)
        resid = np.copy(spec)
        
        # --- Fit negative shift ---
    
        # Get zero-velocity region
        ind, = np.where(np.abs(vel) < vmax)
        imin = ind[0]
        imax = ind[-1]
        lo = imin - freq_switch_offset_chan
        hi = imax - freq_switch_offset_chan
        if hi > 0:
            if lo < 0:
                lo = 0
                imin = freq_switch_offset_chan
            ref = -bestspec[imin:imax]
            if smlen is not None:  # Smooth a little bit
                ref = dln.medfilt(ref,int(smlen))            
            sig = spec[lo:hi]
            # sometimes the relationship is not a simple scalar scaling factor
            #  especially if the zero-velocity region is the near the edge
            negcoef = np.polyfit(ref,sig,2)
            if verbose:
                print('negative shift coef:',negcoef)            
            negmodel = np.polyval(negcoef,ref)
            resid[lo:hi] -= negmodel
        else:
            # completely outside the range
            pass
    
        # --- Fit positive shift ---

        # Get zero-velocity region
        ind, = np.where(np.abs(vel) < 50)
        imin = ind[0]
        imax = ind[-1]
        lo = imin + freq_switch_offset_chan
        hi = imax + freq_switch_offset_chan
        if lo < npix-1:
            if hi > npix-1:
                hi = npix
                imax = npix-freq_switch_offset_chan
            ref = -bestspec[imin:imax]
            if smlen is not None:   # Smooth a little bit
                ref = dln.medfilt(ref,int(smlen))
            sig = spec[lo:hi]
            # sometimes the relationship is not a simple scalar scaling factor
            #  especially if the zero-velocity region is the near the edge
            poscoef = np.polyfit(ref,sig,2)
            if verbose:
                print('positive shift coef:',poscoef)
            posmodel = np.polyval(poscoef,ref)
            resid[lo:hi] -= posmodel
        else:
            # completely outside the range
            pass

        bestspec = resid

    # Make new dictionary
    newsp = {'spec':resid, 'nspec':np.copy(sp['nspec']), 'freq':np.copy(sp['freq']),
             'vel':np.copy(sp['vel']), 'freq_switch_offset':sp['freq_switch_offset'],
             'header':sp['header'].copy(), 'coef':sp['coef'].copy(),
             'poscoef':poscoef, 'negcoef':negcoef}
        
    return newsp

def rawfit(raw,sp):
    """
    Fit the raw baselines with a first-estimate spectrum (with side-lobes cleaned).
    """

    nraw = len(raw)
    npix = len(raw[0]['data'])
    
    # Get initial estimate of the baselines
    data = np.zeros((nraw,npix),float)
    for i in range(nraw):
        data[i,:] = raw[i]['data']
    med = np.nanmin(data,axis=0)
    sig = dln.mad(data,axis=0)
    medsig = np.median(sig)
    resid = np.zeros((nraw,npix),float)
    for i in range(nraw):
        scl = np.nanmedian(data[i,:]/med)
        resid[i,:] = data[i,:] - scl*med
    #resid = data - med.reshape(1,-1)
    bad = (resid > 5*medsig)  # mask positive outliers
    masked_data = np.ma.masked_array(data,mask=bad)
    bline = np.ma.median(masked_data,axis=0)
    
    # Bspline
    x = np.arange(npix)
    p10 = int(npix*0.10)
    p90 = int(npix*0.90)
    w = np.ones(npix)
    vel = getvel(raw[0])
    zerovel, = np.where(np.abs(vel) < 100000)
    w[zerovel] = 0.00001
    bspl = dln.bspline(x[p10:p90],bline[p10:p90],w=w[p10:p90],nquantiles=10,nord=2)
    bspl_model = bspl(x)

    # Loop over the raw spectra and fit them
    refarr = np.zeros((nraw,npix),float)
    calarr = np.zeros((nraw,npix),float)
    for i in range(nraw):
        data1 = data[i,:]
        freq = getfreq(raw[i])
        lo = np.where(np.abs(sp['freq']-freq[0]) < 100)[0][0]
        hi = np.where(np.abs(sp['freq']-freq[-1]) < 100)[0][0]
        spec = sp['spec'][lo:hi+1]
        # use the initial bspline reference to rougly calibrate the raw spectrum
        cal = data1/bspl_model - 1
        bzero = np.nanmedian(cal)
        cal -= bzero
        good = np.isfinite(cal)
        # Find the scaling factor
        sclcoef = np.polyfit(spec[good],cal[good],1)
        # Now fit a line to the residuals
        resid = cal - np.polyval(sclcoef,spec)
        x = np.arange(npix)
        bcoef = np.polyfit(x[good],resid[good],1)
        # Construct a better reference baseline model
        bmodel = bspl_model.copy()
        bmodel *= (1+bzero)                 # scale down the original model
        bmodel *= (1+np.polyval(bcoef,x))   # remove the residual linear baseline
        # Now subtract the actual spectrum from the raw data
        scaled_spec = bmodel * spec*sclcoef[0]
        rawresid = data1 - scaled_spec
        # Refit the bspline
        w = np.ones(npix)
        vel = getvel(raw[i])
        zerovel, = np.where(np.abs(vel) < 100000)
        w[zerovel] = 0.00001
        bspl = dln.bspline(x[good],rawresid[good],w=w[good],nquantiles=10,nord=2)
        best_bmodel = bspl(x)

        # Now iterate
        # While loop
        flag = True
        count = 0
        last_best_bmodel = np.zeros(npix)
        while (flag):
            # calibrate raw spectrum 
            cal = data1/best_bmodel - 1
            bzero = np.nanmedian(cal)
            cal -= bzero
            good = np.isfinite(cal)
            # Find the scaling factor
            sclcoef = np.polyfit(spec[good],cal[good],1)
            # Construct a better reference baseline model
            bmodel = best_bmodel.copy()
            bmodel *= (1+bzero)                 # scale down the original model
            # Now subtract the actual spectrum from the raw data
            scaled_spec = bmodel * spec*sclcoef[0]
            rawresid = data1 - scaled_spec
            # Refit the bspline
            w = np.ones(npix)
            vel = getvel(raw[i])
            zerovel, = np.where(np.abs(vel) < 100000)
            w[zerovel] = 0.00001
            bspl = dln.bspline(x[good],rawresid[good],w=w[good],nquantiles=10,nord=2)
            best_bmodel = bspl(x)
            rdiff = (best_bmodel-last_best_bmodel)/np.nanmedian(best_bmodel)
            maxrdiff = np.max(np.abs(rdiff[good]))
            if maxrdiff < 0.01:
                flag = False
            print(count,maxrdiff)
            count += 1
            last_best_bmodel = best_bmodel.copy()
        # Put final results into the array
        refarr[i,:] = best_bmodel
        calarr[i,:] = data1/best_bmodel - 1

    # Need to deal with the edges!!!

        
    return refarr,calarr

            
def session(filename,tag='_red',outfile=None,scans=None,maskvel=50e3,
            edgetrim=2500,maxiter=5,velrange=[-800e3,800e3],verbose=False):
    """
    Baseline correct a full session of data for a target/map.

    Parameters
    ----------
    filename : str
      GBTIDL output filename.
    tag : str, optional
      Tag to add to final file.  tag='_red' by default.
    outfile : str, optional
      Output filename.  By default the output filename is the same as the
        input filename with TAG added at the end.
    scans : int or list, optional
      List of scans to do.  Default is all.
    maskvel : float, optional
      Velocity around zero-velocity region to mask out.  Default is 30e3 m/s.
    verbose : bool, optional
      Verbose output to the screen.

    Returns
    -------
    Saves the final datacube to the output filename.
    Nothing is returned.


    Example
    -------

    session(filename)

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

    if scans is None:
        scans = np.unique(tab['scan'].data)
    else:
        if type(scans) is not list: scans=[scans]
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
            sp = dointegration(tab1,maskvel=maskvel,maxiter=maxiter,
                               edgetrim=edgetrim,velrange=velrange,verbose=False)
            sp['scan'] = s
            sp['int'] = integ
            sp[lontype] = tab1['crval2'][0]
            sp[lattype] = tab1['crval3'][0]
            finaldata.append(sp)
            
    # Reformat into a large table
    npix = np.max([len(f['spec']) for f in finaldata])
    dt = [('scan',int),('int',int),('data',float,npix),('nspec',int,npix),(lontype,float),(lattype,float)]
    final = np.zeros(len(finaldata),dtype=np.dtype(dt))
    final['data'] = np.nan
    for i in range(len(finaldata)):
        npix1 = len(finaldata[i]['spec'])
        final['scan'][i] = finaldata[i]['scan']
        final['int'][i] = finaldata[i]['int']
        final['data'][i][0:npix1] = finaldata[i]['spec']
        final['nspec'][i][0:npix1] = finaldata[i]['nspec']  # how about exptime?
        final[lontype][i] = finaldata[i][lontype]
        final[lattype][i] = finaldata[i][lattype]        


    #import pdb; pdb.set_trace()

        
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

    # column names must be in all caps
    # HDU1 extname needs to be "SINGLE DISH"
    # gbtgridder expects all of the columns from the single dish data
    
    
    #import pdb; pdb.set_trace()

    #return final,vel
