import numpy as np


class DataContainer(object):
    """
    Create a new data_struct of the requested data type (spectrum or
    continuum) and containing the given array, if supplied.  For
    continuum data, when the data array is supplied, the additional
    pointers (date, utc, mjd, etc) are set to double precision vectors
    of 0s having the same number of elements as arr.
 
    @param arr {in}{optional}{type=integer}{default=u1ndefined} The data
    pointer that this data_struct will hold.  If arr is not provided,
    then the data pointer will point to an undefined variable.

    @keyword spectrum {in}{optional}{default=set} When this is set, a spectrum
    structure will be returned.  That is the default behavior.

    @keyword continuum {in}{optional}{default=unset} When this is set, a
    continuum structure will be returned.  spectrum and continuum are mutually
    exclusive.

    @keyword nocheck {in}{optional}{default=unset} When this is set,
    the input parameter checking is turned off.  Usefull for speed.

    @returns requested data structure of given size or -1 on failure.
    """
    
    def __init__(self,arr=None,spectrum=True,continuum=False,nocheck=False):

        if not nocheck:
            if arr is None or np.array(arr).ndim != 1:
                print('arr must be 1-dimensional')
                return
            if spectrum and continuum:
                print('Only one of spectrum or continuum can be used at a time')
                return

        if continuum:
            return ContinuumDataContainer(arr)
        else:
            return SpectrumDataContainer(arr)

class SpectrumDataContainer(object):

    def __init__(self,arr=None):
        if arr is not None and np.array(arr).ndim == 1:
            self.data = np.empty_like(arr)
        else:
            self.data = None
        self.date = 10*' '                 # date string from date-obs
        self.utc = 0.0                     # seconds on date from date-obs
        self.mjd = 0.0                     # modified julian date from date-obs
        self.frequency_type = 16*' '       # ctype1 in sdfits
        self.reference_frequency = 0.0     # crval1 in sdfits
        self.reference_channel = 0.0       # crpix1 in sdfits
        self.frequency_interval = 0.0      # cdelt1 in sdfits
        self.frequency_resolution = 0.0 
        self.longitude_axis = 0.0          # crval2 in sdfits
        self.latitude_axis = 0.0           # crval3 in sdfits
        self.lst = 0.0                     # calc from date+utc (in UTC)
        self.azimuth = 0.0 
        self.elevation = 0.0  
        self.subref_state = 0
        self.qd_el = np.nan
        self.qd_xel = np.nan
        self.qd_bad = -1
        self.center_frequency = 0.0
        self.line_rest_frequency = 0.0     # restfreq in sdfits
        self.doppler_frequency = 0.0       # dopfreq in sdfits
        self.velocity_definition = 8*' '   # veldef in sdfits
        self.frame_velocity = 0.0          # vframe in sdfits
        # rvsys:0.0   # ?????
        self.source_velocity = 0.0         # velocity in sdfits
        self.zero_channel = 0.0            # zerochan in sdfits
        self.adcsampf = 0.0
        self.vspdelt = 0.0
        self.vsprval = 0.0
        self.vsprpix = 0.0
        self.freq_switch_offset = 0.0      # foffref1 in sdfits
        self.nsave = 0

        if arr is not None and len(arr) > 0:
            self.data = arr
        else:
            self.data = np.empty(0)

        
class ContinuumDataContainer(object):

    def __init__(self,arr=None):
        if arr is not None and len(arr)>0:
            # initialize other arrays to 0's of same length as arr
            zeros = np.zeros(len(arr),float)
            self.date = np.zeros(len(arr),(str,10))  # date string from date-obs
            self.utc = zeros                   # seconds on date from date-obs
            self.mjd = zeros                   # modified julian date from date-obs
            self.longitude_axis = zeros        # crval2 in sdfits
            self.latitude_axis = zeros         # crval3 in sdfits
            self.lst = zeros                   # calc from date+utc (in UTC)
            self.azimuth = zeros
            self.elevation = zeros
            # except subref_state, all 1s
            self.subref_state = np.ones(len(arr),int)
            # and qd_el, qd_xel are all NaNs
            self.qd_el = zeros+np.nan
            self.qd_xel = zeros+np.nan
            # and qd_bad is all -1        
            self.qd_bad = np.zeros(len(arr),int)-1
            self.data = arr
        else:
            self.date = None               # date string from date-obs
            self.utc = None                # seconds on date from date-obs
            self.mjd = None                # modified julian date from date-obs
            self.longitude_axis = None     # crval2 in sdfits
            self.latitude_axis = None      # crval3 in sdfits
            self.lst = None                # calc from date+utc (in UTC)
            self.azimuth = None
            self.elevation = None
            self.subref_state = None
            self.qd_el = None
            self.qd_xel = None
            self.qd_bad = None
            self.data = None
