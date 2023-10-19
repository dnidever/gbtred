import numpy as np
from dlnpyutils import utils as dln
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve

def cubize(tab):
    """ Return a datacube."""

    scans = np.unique(tab['scan'])
    nscan = len(scans)
    ints = np.unique(tab['int'])
    nint = len(ints)
    npix = tab['data'].shape[1]
    cube = np.zeros((nscan,nint,npix),float)

    # scan is latitude
    # int is longitude
    glon = np.zeros((nscan,nint),float)
    glat = np.zeros((nscan,nint),float)    
    for i,s in enumerate(scans):
        ind, = np.where(tab['scan']==s)
        if i % 2 == 1:
            ind = np.flip(ind)
        cube[i,:,:] = tab['data'][ind,:]
        glon[i,:] = tab['glon'][ind]
        glat[i,:] = tab['glat'][ind]        

    return cube,glon,glat

def uncubize(cube):
    """ Uncubize back to original 2D format."""
    nx,ny,npix = cube.shape
    uncube = np.zeros((nx*ny,npix),float)
    count = 0
    for i in range(nx):
        ind = np.arange(ny)
        if i % 2 == 1:
            ind = np.flip(ind)
        uncube[count:count+ny,:] = cube[i,ind,:]
        count += ny
    return uncube
        
def make_kernel(data,fwhm,truncate=4.0):
    """ Make 2D Gaussian Kernele for gsmooth()"""
    xsize = np.ceil(fwhm/2.35*truncate*2)
    if dln.size(fwhm)==1:
        if xsize % 2 == 0: xsize+=1   # must be odd            
        kernel = Gaussian2DKernel(fwhm/2.35,x_size=xsize)
    else:
        if xsize[0] % 2 == 0: xsize[0]+=1   # must be odd
        if xsize[1] % 2 == 0: xsize[1]+=1   # must be odd              
        kernel = Gaussian2DKernel(fwhm[0]/2.35,fwhm[1]/2.35,x_size=xsize)
    return kernel
        
def gsmooth2d(data,kernel,mask=None,boundary='extend',fill=0.0):
    # astropy.convolve automatically ignores NaNs
    return convolve(data, kernel.array, mask=mask, boundary=boundary, fill_value=fill)

def psmooth(cube,smlen):
    """ Gaussian smooth spatially."""
    # Assuming the first two dimensions are the spatial ones
    sh = cube.shape
    npix = sh[2]
    smcube = np.zeros(cube.shape,float)
    kernel = make_kernel(cube[:,:,0],smlen)
    for i in range(npix):
        smcube[:,:,i] = gsmooth2d(cube[:,:,i],kernel)
    return smcube
        
def vsmooth(cube,smlen):
    """ Gaussian smooth in velocity."""
    # Assuming the first two dimensions are the spatial ones
    nx,ny,npix = cube.shape
    smcube = np.zeros(cube.shape,float)
    for i in range(nx):
        for j in range(ny):
            smcube[i,j,:] = dln.gsmooth(cube[i,j,:],smlen)
    return smcube

def pvsmooth(cube,psmlen,vsmlen):
    """ Position and velocity Gaussian smoothing."""
    return vsmooth(psmooth(cube,psmlen),vsmlen)
