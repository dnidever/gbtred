import os
import numpy as np

def dofreqswitch(sigwcal, sig, refwcal, ref, smoothref, tsys=None,
                 tau=None, tcal=None, sigResult=None, refResult=None):
    """
    This procedure calibrates a single integration from a frequency
    switched scan.

    <p>The expected 4 spectra that are used here are the signal with no
    cal, the signal with cal, the reference with no cal and the
    reference with cal.  
 
    <ul><li><a href="dototalpower.html">dototalpower</a> is used to get
    the total power for the two signal spectra and for the two reference
    spectra.  
    <li>These are then combined using
    <a href="dosigref.html">dosigref</a> to get the calibrated results.
    <ul><li>sigresult is done using the signal total power as
    the "signal" and the reference total power as the "reference"
    <li>refresult is done using the reference reference total power as
    the "signal" and the signal total power as the "reference".
    </ul>
    <li>See dototalpower and dosigref for details about
    each step in the process of producing the two results.
    </ul>

    <p><a href="dcfold.html">dcfold</a> can then be used to combine
    sigresult and refresult to produce a folded result. That step does
    not happen here.

    <p>The user can optionally over-ride the reference system
    temperature calculated in dototalpower and used in dosigref by
    supplying a value for the tsys and tau keywords here.  tsys is the
    system temperate at tau=0.  If the user supplies this keyword, tsys
    is first adjusted to the elevation of the reference spectrum : 
    <pre>
    tsys_used = tsys*exp(tau/sin(elevation).  
    </pre>
    If tau is not supplied, then the <a href="get_tau.html">get_tau</a>
    function is used, using the reference observed_frequency to arrive
    at a very rough guess as to the zenith opacity, tau.  Users are
    encouraged to supply tau when they also supply tsys to improve the
    accuracy of this calculation. The adjusted tsys then becomes the
    reference spectrum's tsys value for use in dosigref.

    <p>The units of sigresult and refresult are "Ta".  Use 
    <a href="dcsetunits.html">dcsetunits</a> to change these units to
    something else.

    <p>This is used primarily by <a href="../guide/getfs.html">getfs</a>
    and this code does almost no argument checks or sanity checks.  The
    calling routine is expected to check that the 4 input spectra are
    compatible (all are valid data containers and all have the same
    number of data points).

    <p>It is the responsibility of the caller to ensure that sigResult
    and refResult are freed using <a href="data_free.html">data_free</a>
    when their use is finished (i.e. at the end of all anticipated calls
    to this function before returning to the calling level).  Failure to
    do that will result in memory leaks.  It is not necessary to free
    these data containers between consecutive calls to this function at
    the same IDL level (e.g. inside the same procedure).

    @param sigwcal {in}{required}{type=spectrum} An uncalibrated
    spectrum from the signal phase with the cal on.
    @param sig {in}{required}{type=spectrum} An uncalibrated
    spectrum from the signal phase with the cal off.
    @param refwcal {in}{required}{type=spectrum} An uncalibrated
    spectrum from the reference phase with the cal on.
    @param smoothref {in}{optional}{type=integer} Boxcar smooth width
    for reference spectrum.  No smoothing if not supplied or if value is
    less than or equal to 1.
    @keyword tsys {in}{optional}{type=float} tsys at zenith, this is
    converted to a tsys at the observed elevation.  If not suppled, the
    tsys for each integration is calculated as described elsewhere.
    @keyword tau {in}{optional}{type=float} tau at zenith, if not
    supplied, it is estimated using <a href="../toolbox/get_tau.html">get_tau</a>
    tau is only used when the requested units are other than the default
    of Ta and when a user-supplied tsys value at zenith is to be used.
    @keyword sigResult {out}{required}{type=spectrum} The result when
    using the signal phases as "sig" in dosigref.
    @keyword refResult {out}{optional}{type=spectrum} This result when
    using the reference phases as "sig" in dosigref.
    """
    
    dototalpower(sigTP, sig, sigwcal, tcal=tcal)
    dototalpower(refTP, ref, refwcal, tcal=tcal)

    # is there a user-supplied tsys
    if tsys is not None:
        # correct this for elevation
        # both data containers matter here
        if tau is None:
            thistauRef = get_tau(refTP.observed_frequency / 1.0e9)
            thistauSig = get_tau(sigTP.observed_frequency / 1.0e9)
        else:
            thistauRef = tau
            thistauSig = tau
        refTP.tsys = tsys * np.exp(thistauRef / np.sin(refTP.elevation))
        sigTP.tsys = tsys * np.exp(thistauSig / np.sin(sigTP.elevation))

    dosigref(sigResult, sigTP, refTP, smoothref)
    dosigref(refResult, refTP, sigTP, smoothref)
    
    # calculate freq_switch_offset - just use the
    # difference at channel 0 - assumes both spectra have
    # the same default frequency axis
    sigResult.freq_switch_offset = chanToFreq(refResult, 0.0) - chanToFreq(sigResult, 0.0)
    sigResult.tsysref = refResult.tsys
    refResult.freq_switch_offset = chanToFreq(sigResult, 0.0) - chanToFreq(refResult, 0.0)
    refResult.tsysref = sigResult.tsys

    data_free(sigTP)
    data_free(refTP)


def dototalpower(result, sig_off, sig_on, tcal=None):
    """
    This procedure calibrates a single integration from a total power
    scan.

    <p>The result is the average of the data in the two data
    containers:
    <pre>
    (*result.data_ptr) = (*sig_off.data_ptr + *sig_on.data_ptr)/2.0
    </pre>
    The tsys in the result is meanTsys as calculated by
    <a href="../toolbox/dcmeantsys.html">dcmeantsys</a>.  The
    integration and exposure times in the result are the sum of those
    two times from each data container. All other header parameters in
    the result are copies of their values in the sig_off 
    spectrum.  dcmeantsys uses the mean_tcal value found in the sig_off
    data container unless the user supplies a tcal value using the tcal
    keyword.  The mean_tcal value in result will reflect the actual
    tcal value used (as resuted by dcmeantsys).

    <p>This simple routine is designed to be called from a more
    complicated routine like gettp.  This does not check the arguments
    for consistency or type.
    
    <p>It is the responsibility of the caller to ensure that result
    is freed using <a href="data_free.html">data_free</a>
    when it is no longer needed (i.e. at the end of all anticipated calls
    to this function before returning to the calling level).  Failure to
    do that will result in memory leaks.  It is not necessary to free
    these data containers between consecutive calls to this function at
    the same IDL level (e.g. inside the same procedure).
    
    @param result {out}{required}{type=spectrum} The result as described
    above.
    @param sig_off {in}{out}{required}{type=spectrum} An uncalibrated
    spectrum with no cal signal.
    @param sig_on {in}{required}{type=spectrum} An uncalibrated
    spectrum with a cal signal.
    @keyword tcal {in}{optional}{type=float} A scalar value for the cal
    temperature (K).  If not supplied. sig_off.mean_tcal will be used.
    """

    data_copy(sig_off, result)
    result.tsys, used_tcal = dcmeantsys(sig_off, sig_on, tcal=tcal, used_tcal=True)
    result.mean_tcal = used_tcal

    # ignore float underflows
    oldExcept = np.seterr(all='ignore')

    result.data_ptr = (sig_off.data_ptr + sig_on.data_ptr) / 2.0

    # clear them
    np.check_math(mask=32)

    # reset except state
    np.seterr(**oldExcept)

    result.exposure = sig_off.exposure + sig_on.exposure
    result.duration = sig_off.duration + sig_on.duration


def dcmeantsys(dc_nocal, dc_withcal, tcal=None, used_tcal=None):
    """
    Calculate the mean Tsys using the data from two spectral line data
    containters, one with the CAL on and one with CAL off.
    
    <pre>
    mean_tsys = tcal * mean(nocal) / (mean(withcal-nocal)) + tcal/2.0
    </pre>
    where nocal and withcal are the data values from dc_nocal and
    dc_withcal and tcal is as described below.

    <ul>
    <li>The outer 10% of all channels in both data containers are
    ignored.
    <li>Blanked data values are ignored.
    <li>The tcal value used here comes from the dc_nocal data container
    unless the user supplies a value in the tcal keyword.
    <li>The tcal value actually used is returned in used_tcal.
    </ul>
    
    <p>This is used by the GUIDE calibration routines and is
    encapsulated here to ensure consistency.  

    @param dc_nocal {in}{required}{type=spectrum data container} The
    data with no cal signal.
    @param dc_withcal {in}{required}{type=spectrum data container} The
    data with a cal signal.
    @keyword tcal {in}{optional}{type=float} A scalar value for the cal
    temperature (K).  If not supplied. dc_nocal.mean_tcal will be used.
    @keyword used_tcal {out}{optional}{type=float} The tcal value
    actually used.
    """

    
    if tcal is None or len(tcal) > 1:
        if len(tcal) > 1:
            print('Vector tcal is not yet supported, sorry. Ignoring user-supplied tcal.')
        used_tcal = dc_nocal.mean_tcal
    else:
        used_tcal = tcal[0]

    # Use the inner 80% of data to calculate mean Tsys
    nchans = len(dc_nocal.data_ptr)
    pct10 = int(nchans / 10)
    pct90 = nchans - pct10

    # Ignore math errors here, underflow is fairly common
    oldExcept = np.seterr(all='ignore')

    meanTsys = np.nanmean(dc_nocal.data_ptr[pct10:pct90], dtype=np.float64) / \
               np.nanmean(dc_withcal.data_ptr[pct10:pct90] - dc_nocal.data_ptr[pct10:pct90], dtype=np.float64) * \
               used_tcal + used_tcal / 2.0

    # Clear them, but only the underflow
    np.check_math(mask=32)

    # Return to previous state
    np.seterr(**oldExcept)

    return meanTsys

def data_free(data_struct):
    """
    Free the data pointer in a data structure.  This should only be
    used when the data structure is no longer necessary since it leaves
    the data pointer in an invalid state.
 
    @param data_struct {in}{out}{required}{type=data_container_struct} The
    struct to free.
    """
    
    if data_valid(data_struct, name=name) == -1:
        print('data_struct must be a valid continuum or spectrum structure')
        return

    for i in range(n_elements(data_struct)):
        # Both have data_ptr
        if ptr_valid(data_struct[i].data_ptr):
            ptr_free(data_struct[i].data_ptr)

        # Continuum has more
        if name == 'CONTINUUM_STRUCT':
            if ptr_valid(data_struct[i].date):
                ptr_free(data_struct[i].date)
            if ptr_valid(data_struct[i].utc):
                ptr_free(data_struct[i].utc)
            if ptr_valid(data_struct[i].mjd):
                ptr_free(data_struct[i].mjd)
            if ptr_valid(data_struct[i].longitude_axis):
                ptr_free(data_struct[i].longitude_axis)
            if ptr_valid(data_struct[i].latitude_axis):
                ptr_free(data_struct[i].latitude_axis)
            if ptr_valid(data_struct[i].lst):
                ptr_free(data_struct[i].lst)
            if ptr_valid(data_struct[i].azimuth):
                ptr_free(data_struct[i].azimuth)
            if ptr_valid(data_struct[i].elevation):
                ptr_free(data_struct[i].elevation)
            if ptr_valid(data_struct[i].subref_state):
                ptr_free(data_struct[i].subref_state)
            if ptr_valid(data_struct[i].qd_el):
                ptr_free(data_struct[i].qd_el)
            if ptr_valid(data_struct[i].qd_xel):
                ptr_free(data_struct[i].qd_xel)
            if ptr_valid(data_struct[i].qd_bad):
                ptr_free(data_struct[i].qd_bad)

    return



def data_copy(in_data, out_data):
    """
    Copy the data container from in to out.

    This procedure can be used to generate a new data container, or to
    copy one data container into an existing data container.  However, it
    cannot be used to copy into one of the global data containers.
    To copy a data container stored as a local variable into a global
    data container, use the procedure set_data_container.
 
    @param in {in}{required}{type=data_container_struct} The data container
    copied.  This can be identified by a local valiable or it can be
    a global data container, such as !g.s[0]

    @param out {out}{required}{type=data_container_struct} The data container to receive
    the copy.  This can be a local variable, but NOT a global data container.

    @examples
    <pre>
    get, mc_scan=22, cal='F', sig='T', pol='XX', if_num=1, int=1  sig
    data_copy,!g.s[0],spec
    a = getdcdata(spec)
    a = a * 2.0
    setdcdata,spec,a
    show,spec
    data_free, spec   clean up memory
    </pre>

    @uses <a href="data_valid.html">data_valid</a>
    @uses <a href="data_free.html">data_free</a>
    """
    
    if data_valid(in_data, name=name) == -1:
        print('in must be a valid continuum or spectrum structure')
        return

    # Preserve the pointers in out_data
    outDataPtr = -1
    outDatePtr = -1
    outUtcPtr = -1
    outMjdPtr = -1
    outLongPtr = -1
    outLatPtr = -1
    outLstPtr = -1
    outAzPtr = -1
    outElPtr = -1
    outSubrefPtr = -1
    outQdElPtr = -1
    outQdXelPtr = -1
    outQdBadPtr = -1

    if data_valid(out_data, name=outname) >= 0:
        if outname != name:
            # Not the same type, free out_data
            data_free(out_data)
        else:
            # If one is valid, they are all valid
            outDataPtr = out_data.data_ptr
            if name == 'CONTINUUM_STRUCT':
                outDatePtr = out_data.date
                outUtcPtr = out_data.utc
                outMjdPtr = out_data.mjd
                outLongPtr = out_data.longitude_axis
                outLatPtr = out_data.latitude_axis
                outLstPtr = out_data.lst
                outAzPtr = out_data.azimuth
                outElPtr = out_data.elevation
                outSubrefPtr = out_data.subref_state
                outQdElPtr = out_data.qd_el
                outQdXelPtr = out_data.qd_xel
                outQdBadPtr = out_data.qd_bad

    # Copy everything, including pointers
    out_data = in_data

    # Restore the out_data pointers
    if not ptr_valid(outDataPtr):
        # None of them are valid
        out_data.data_ptr = ptr_new(allocate_heap=True)
        if name == 'CONTINUUM_STRUCT':
            out_data.date = ptr_new(allocate_heap=True)
            out_data.utc = ptr_new(allocate_heap=True)
            out_data.mjd = ptr_new(allocate_heap=True)
            out_data.longitude_axis = ptr_new(allocate_heap=True)
            out_data.latitude_axis = ptr_new(allocate_heap=True)
            out_data.lst = ptr_new(allocate_heap=True)
            out_data.azimuth = ptr_new(allocate_heap=True)
            out_data.elevation = ptr_new(allocate_heap=True)
            out_data.subref_state = ptr_new(allocate_heap=True)
            out_data.qd_el = ptr_new(allocate_heap=True)
            out_data.qd_xel = ptr_new(allocate_heap=True)
            out_data.qd_bad = ptr_new(allocate_heap=True)
    else:
        # All of them are valid
        out_data.data_ptr = outDataPtr
        if name == 'CONTINUUM_STRUCT':
            out_data.date = outDatePtr
            out_data.utc = outUtcPtr
            out_data.mjd = outMjdPtr
            out_data.longitude_axis = outLongPtr
            out_data.latitude_axis = outLatPtr
            out_data.lst = outLstPtr
            out_data.azimuth = outAzPtr
            out_data.elevation = outElPtr
            out_data.subref_state = outSubrefPtr
            out_data.qd_el = outQdElPtr
            out_data.qd_xel = outQdXelPtr
            out_data.qd_bad = outQdBadPtr

    # Now copy the data values into the pointers
    # All pointers should be valid here
    if data_valid(in_data) == 0:
        if size(out_data.data_ptr, /type) != 0:
            # No other way to do this, I think
            # If one is not undefined, they are all defined
            # They need to be undefined here
            ptr_free(out_data.data_ptr)
            out_data.data_ptr = ptr_new(allocate_heap=True)
            if name == 'CONTINUUM_STRUCT':
                ptr_free(out_data.date)
                ptr_free(out_data.utc)
                ptr_free(out_data.mjd)
                ptr_free(out_data.longitude_axis)
                ptr_free(out_data.latitude_axis)
                ptr_free(out_data.lst)
                ptr_free(out_data.azimuth)
                ptr_free(out_data.elevation)
                ptr_free(out_data.subref_state)
                ptr_free(out_data.qd_el)
                ptr_free(out_data.qd_xel)
                ptr_free(out_data.qd_bad)
                out_data.date = ptr_new(allocate_heap=True)
                out_data.utc = ptr_new(allocate_heap=True)
                out_data.mjd = ptr_new(allocate_heap=True)
                out_data.longitude_axis = ptr_new(allocate_heap=True)
                out_data.latitude_axis = ptr_new(allocate_heap=True)
                out_data.lst = ptr_new(allocate_heap=True)
                out_data.azimuth = ptr_new(allocate_heap=True)
                out_data.elevation = ptr_new(allocate_heap=True)
                out_data.subref_state = ptr_new(allocate_heap=True)
                out_data.qd_el = ptr_new(allocate_heap=True)
                out_data.qd_xel = ptr_new(allocate_heap=True)
                out_data.qd_bad = ptr_new(allocate_heap=True)
    else:
        *out_data.data_ptr = *in_data.data_ptr
        if name == 'CONTINUUM_STRUCT':
            *out_data.date = *in_data.date
            *out_data.utc = *in_data.utc
            *out_data.mjd = *in_data.mjd
            *out_data.longitude_axis = *in_data.longitude_axis
            *out_data.latitude_axis = *in_data.latitude_axis
            *out_data.lst = *in_data.lst
            *out_data.azimuth = *in_data.azimuth
            *out_data.elevation = *in_data.elevation
            *out_data.subref_state = *in_data.subref_state
            *out_data.qd_el = *in_data.qd_el
            *out_data.qd_xel = *in_data.qd_xel
            *out_data.qd_bad = *in_data.qd_bad

    return
