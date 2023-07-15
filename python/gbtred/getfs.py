import numpy as np
from . import toolbox as tb


def getIftabIndices(loc, nif, nfd, npl):
    """
    Private function to return the array indices in a 3D iftable array
    given the array dimensions.

    Used instead of array_indices because of the way IDL tosses out
    single-element arrays whenever possible, making it damn difficult to
    write general purpose code.  This routine always returns 3 values.

    The length of the 3rd axis is not important here.  There is no
    checking loc for validity (the only reason this might care about
    that value).

    IDL arrays are stored in fortran order.

    @param loc {in}{required}{type=integer} Vector of locations into a
    3D array described by the other parmeters.
    @param nif {in}{required}{type=integer} Length of first axis.
    @param nfd {in}{required}{type=integer} Length of second axis.

    @returns (3,n_elements(loc)) array, one 3-D vector giving the
    coordinates into a 3D array described by nif, nfd, npl for each
    element of loc.
    """
    
    nloc = len(loc)
    result = np.zeros((3, nloc), dtype=int)

    for i in range(nloc):
        thisLoc = loc[i]
        
        ifnum = thisLoc % nif
        fdnum = ((thisLoc - ifnum) // nif) % nfd
        plnum = (thisLoc - ifnum - fdnum * nfd) // (nif * nfd)

        result[:, i] = [ifnum, fdnum, plnum]

    return result


def scan_info(scan, filename, keep=False, quiet=False):
    """
    Get scan information structure from the appropriate I/O object.
    
    <p>This uses !g.line to determine which I/O object to get the
    information from, unless keep is set, in which case it gets it from
    the keep file.

    <p>Note that this may return more than one structure (an array of
    structures would be returned).  This can happen if the same scan
    number appears in the data with more than one timestamp.  The
    instance and timestamp keywords of the various data retrieval
    procedures (getscan, get, getnod, getps, etc) can be used to
    differentiate between these duplicate scans.  The instance keyword
    in those cases corresponds to the element number of the array
    returned by scan_info.  The timestamp keyword corresponds to the
    timestamp field in the scan_info structure.

    <p>The data corresponding to a given instance are always found in
    consecutive records.  The index_start and nrecords fields can be
    used to get just the data associated with a specific instance.  See
    the examples.

    The fields in the returned structure are:
    <UL>
    <LI> SCAN, the scan number, long integer
    <LI> PROCSEQN, the procedure sequence number, long integer
    <LI> PROCEDURE, the procedure name, string
    <LI> TIMESTAMP, the timestamp associated with this scan, string
    <LI> FILENAME, the name of the file where this scan was found
    <LI> N_INTEGRATION, the number of integrations, long integer.  Note
    that different samplers may have different numbers of integrations.
    This value is the number of unique times (MJD) found for this scan.
    Use N_SAMPINTS for the number of integrations from a given sampler.
    <LI> N_FEEDS, the number of feeds, long integer
    <LI> FDNUMS, the specific FDNUM values for this scan.
    <LI> N_IFS, the number of IFs, long integer
    <LI> IFNUMS, the specific IFNUM values for this scan.
    <LI> IFTABLE, a 3-axis array indicating which combinations of ifnum,
    fdnum and plnum are present in the data.  The first axis is ifnum,
    the second axis is fdnum and the 3rd is plnum.  Combinations with
    data have 1 in this array.  Combinations without data have a 0.
    Note: IDL removes any trailing axis of length 1 (degenerate) so care
    must be taken when using the shape of this array.  E.g. if there is
    no third axis, then there is only one plnum, plnum=0.
    <LI> N_SWITCHING_STATES, the number of switching states, long integer
    <LI> N_CAL_STATES, the number of cal switching states, long integer
    <LI> N_SIG_STATES, the number of sig switching states, long integer
    <LI> N_WCALPOS, the total number of unique WCALPOS values in this
    scan. (spectral line data only)
    <LI> WCALPOS, a vector giving the list of unique WCALPOS
    (WBand receiver calposition) values for this scan. (spectral line
    data only)
    <LI> N_POLARIZATIONS, the number of unique polarizations, long
    integer, will always be less then or equal to 4.
    <LI> POLARIZATIONS, a vector containing the actual
    polarizations, string (unused elements are the null string).
    <LI> PLNUMS, vector containing the PLNUM values
    corresponding to each POLARIZATION, long integer (unused elements
    are -1)
    <LI> FEEDS, a vector containing the unique feed ids, long
    integer (unused elements are -1)
    <LI> BANDWIDTH, a vector containing the unique bandwidths, one for each IF
    (Hz)
    <LI> INDEX_START, the starting index number for this scan.
    <LI> NRECORDS, the total number of records in this scan.
    <LI> N_SAMPLERS, the total number of unique sampler names in this scan.
    <LI> SAMPLERS, the list of unique sampler names.
    <LI> N_SAMPINTS, the number of integrations for each sampler.
    <LI> N_CHANNELS, the number of channels in each spectrum for each sampler.
    </UL>

    @param scan {in}{required}{type=integer} Scan number to get
    information on.

    @param filename {in}{optional}{type=string} Limit the search for
    matching scans to a specific file.  If omitted, scans are found in
    all files currently opened through filein (a single file) or dirin
    (possibly multiple files).

    @keyword keep {in}{optional}{type=boolean} If set, the scan
    information comes from the keep file.

    @keyword quiet {in}{optional}{type=boolean} When set, suppress most
    error messages.  Useful when being used within another procedure.

    @keyword count {in}{optional}{type=integer} Returns the number of
    elements of the returned array of scan_info structures.

    @returns Scan information structure.  Returns -1 on error.

    @examples
    Get all of the data associated with one scan.
    <pre>
    a = scan_info(65)
    indx = lindgen(a.nrecords) + a.index_start
    d65 = getchunk(index=indx)
    ... do stuff with d65, but don't forget to free it when done
    data_free, d65
    </pre>
    <p>
    Find paired scan's and their info structure (e.g. position switched)
    <pre>
    a = scan_info(65)
    b = find_paired_info(a)
    </pre>
    """
    
    count = 0

    if len(scan) == 0:
        print('scan_info')
        return -1

    result = -1
    count = 0

    if keep:
        if not lineoutio.is_data_loaded():
            result = lineoutio.get_scan_info(scan, filename, count=count, quiet=quiet)
        elif not quiet:
            print('There is no data in the keep file')
    else:
        if not line:
            if not lineio.is_data_loaded():
                result = lineio.get_scan_info(scan, filename, count=count, quiet=quiet)
            elif not quiet:
                print('There is no data in the line file')
        else:
            if not contio.is_data_loaded():
                result = contio.get_scan_info(scan, filename, count=count, quiet=quiet)
            elif not quiet:
                print('There is no data in the continuum file')

    if count <= 0 and not quiet:
        print('That scan was not found')

    return result


def find_scan_info(scan, timestamp=None, instance=None, filename=None):
    """
    Find a scan info from the current line filein matching the given scan,
    file, timestamp, and instance values.  The matching scan_info is
    returned.  See <a href="scan_info.html">scan_info</a> for more
    information on the returned structure.

    <p>This is used by all of the standard calibration routines to get
    the scan_info for the requested scan.  It is encapsulated here to
    make it easy to adapt and understand those calibration routines.

    If there was a problem, the return value will not be structure (it
    will be -1).

    Because this is designed to be called from another routine, any
    error messages are displayed using the prefix appropriate to the
    calling routine.

    @param scan {in}{required}{type=integer} Scan number to get
    information on.  This must be provided unless timestamp is provided.
    @keyword timestamp {in}{optional}{type=string} The M&C timestamp associated
    with the desired scan. When supplied, scan and instance are ignored.
    @keyword instance {in}{optional}{type=integer} Which occurence
    of this scan should be used.  Default is 0.
    @keyword filename {in}{optional}{type=string} Limit the search for
    matching scans to a specific file.  If omitted, scans are found in
    all files currently opened through filein (a single file) or dirin
    (possibly multiple files).
    @returns A single scan_info structure.  Returns -1 if a match can
    not be found.

    @uses <a href="../toolbox/select_data.html">select_data</a>
    @uses <a href="scan_info.html">scan_info</a>
    """
    
    if timestamp is not None:
        if len(timestamp) > 1:
            print('Only one timestamp can be specified')
            return -1

        recs = select_data(lineio, count=count, timestamp=timestamp, filename=filename)
        if count <= 0:
            if file is not None:
                print('No data having that timestamp is available in file=' + filename)
            else:
                print('No data having that timestamp is available.')
            return -1

        thisScan = lineio.get_index_values('SCAN', index=recs[0])
        if filename is not None:
            thisFile = lineio.get_index_values('FILE', index=recs[0])
            info = scan_info(thisScan, thisFile, count=count)
        else:
            info = scan_info(thisScan, count=count)

        if count < 0:
            print('Unexpectedly did not find a matching scan - this should never happen')
        theseTimes = info.timestamp
        thisInstance = np.where(theseTimes == timestamp)[0]
        if len(thisInstance) < 0:
            print('Unexpectedly did not find matching timestamp in scan_info record - this should never happen')
        info = info[thisInstance]
    else:
        info = scan_info(scan, filename, count=count, quiet=True)
        if count <= 0:
            if filename is not None:
                print('That scan is not available in file=' + filename)
            else:
                print('That scan is not available.')
            return -1

        if instance is not None:
            if instance >= count:
                print('Requested instance does not exist, it must be <', count)
                return -1
            info = info[instance]
        else:
            if count > 1:
                print('More than one scan found, using the first one (instance=0)')
            info = info[0]

    return info


def check_calib_args(scan, refscan, intnum=None, ifnum=None, plnum=None, fdnum=None,
                     sampler=None, eqweight=None, units=None, bswitch=None, quiet=None,
                     keepints=None, useflag=None, skipflag=None, instance=None, filename=None,
                     timestamp=None, refinstance=None, reffile=None, reftimestamp=None,
                     checkref=None, tau=None, ap_eff=None, twofeeds=None, sig_state=None):
    """
    This function is used by the standard calibration routines to handle
    some argument checking and to assign default values to keywords
    when not provided by the user.

    <p>Encapsulating these here should make it easier for users to
    adapt a calibration routine to do what they want it to do.

    <p>Since the calibration routines all only work for line data,
    GBTIDL must currently be in line mode when this routine is called.
    If it is in continuum mode, that is an error and this function will
    return -1.  In addition, there must already be line data opened
    using either filein or dirin.
    
    <p>The argument descriptions here refer to what this routine checks
    for, not what the argument means.  For the meaning of a specific
    argument, see the calibration routine in question.  Type checking is
    only done for string keywords.

    <p>Because this routine is designed to be called by another routine,
    errors are reported such that the message prefix is the name of the
    calling routine.  Users are least likely to be confused by those
    messages.
 
    <p>A warning is printed if tau or ap_eff are specified and the units
    value (explicit or implied) means tau or ap_eff are not used
    (e.g. the default units 'Ta' do not need tau or ap_eff and so if
    they are provided, a warning to that effect is printd).  This not
    considered a severe problem and processing continues.  This can be
    turned off if the quiet keyword is set.

    <p>If there was a severe problem, the return value is 0 (false) and
    the calling routine should exit at that point.  If the arguments are
    all okay then the return value is 1 and any defaults are returned in
    a structure in the defaults keyword value.

    <p>If sampler is supplied then all 3 of ifnum, plnum, and fdnum must
    not be supplied.  The returned values for these 3 are all -1,
    implying that sampler should be used.

    <p>If ifnum, fdnum, or plnum are not supplied, the lowest valid
    value with data is chosen.  This value is picked by first setting,
    ifnum, then fdnum, and finally plnum (using any user-supplied values
    first).  If there is no valid data using the user-supplied values
    then <a href="showiftab.html">showiftab</a> is used to display the set of valid values and the
    return value is -1.

    @param scan {in}{optional}{type=integer} If scan is not supplied,
    then a valid timestamp keyword must be supplied. No default supplied.
    @param refscan {in}{optional}{type=integer} Ignored unless checkref
    is true.  If refscan is not supplied, then a valid reftimestamp
    keyword must be supplied.  No default supplied.
    @keyword intnum {in}{optional}{type=integer} Must be >= 0.
    @keyword ifnum {in}{optional}{type=integer} Must be >= 0. Defaults
    as described above.
    @keyword plnum {in}{optional}{type=integer} Kust be >= 0. Defaults
    as described above.
    @keyword fdnum {in}{optional}{type=integer} Must be >= 0. Defaults
 
    @keyword sampler {in}{optional}{type=string} Must be non-empty.
    Defaults to '' (empty, unspecified).  When set, the returned ifnum,
    plnum, and fdnum values are all -1.
    @keyword eqweight {in}{optional}{type=boolean}
    @keyword units {in}{optional}{type=string} Must be one of
    "Ta","Ta*", or "Jy".
    @keyword bswitch {in}{optional}{type=integer} Must be 0, 1 or 2.
    Defaults to 0.
    @keyword quiet {in}{optional}{type=boolean}
    @keyword keepints {in}{optional}{type=boolean}
    @keyword useflag {in}{optional}{type=boolean} Only one of useflag
    and skipflag can be set.
    @keyword skipflag {in}{optional}{type=boolean} Only one of useflag
    and skipflag can be set.
    @keyword instance {in}{optional}{type=integer} Must be >=
    0. Defaults to 0.
    @keyword filename {in}{optional}{type=string}
    @keyword timestamp {in}{optional}{type=string} If scan is not
    supplied, then a valid timestamp keyword must be supplied.
    @keyword refinstance {in}{optional}{type=integer} Ignored unless
    checkref is true.  Must be >= 0.  Defaults to 0.
    @keyword reffile {in}{optional}{type=string} Ignored unless checkref
    is true.
    @keyword reftimestamp {in}{optional}{type=string} Ignored unelss
    checkref is true.  If refscan is not supplied, then a valid
    reftimestamp keyword must be supplied.
    @keyword checkref {in}{optional}{type=boolean} Check refscan and the
    ref* keywords?
    @keyword tau {in}{optional}{type=float} Warning if tau is set and
    units is 'Ta' or unset.
    @keyword ap_eff {in}{optional}{type=float} Warning if units is not
    'Jy'.
    @keyword twofeeds {in}{optional}{type=boolean} When set, fdnum is
    assumed to be a tracking feed number and it is not influenced by any
    value that sampler might have.
    @keyword sig_state {in}{optional}{type=integer} Used for sig_state
    selection.  When set it must be 0 or 1.  Returned value is -1 if
    unset or out of bounds.
    @keyword ret {out}{required}{type=structure} The values to use for
    ifnum, plnum, fdnum, instance, and bswitch taking into account the defaults
    as described here.  This is done so that the values of the calling
    arguments are not altered by this function.
    @keyword info {out}{required}{type=structure} The scan info structure
    associated with the scan, timestamp, instance and file arguments as
    given.  This will not be a structure if there was a problem.
    """
    result = 0

    # basic checks
    if not line:
        print('This does not work in continuum mode, sorry.')
        return result,None,None

    if useflag is not None and skipflag is not None:
        print('Useflag and skipflag cannot be used at the same time')
        return result,None,None

    if not lineio.is_data_loaded():
        print('No line data is attached yet, use filein, dirin, online or offline')
        return result,None,None

    if not scan and not timestamp:
        print('The scan number is required unless a timestamp is provided.')
        return result,None,None

    # string argument type checks
    if filename is not None:
        if not isinstance(filename, str):
            print('File must be a string')
            return result,None,None

    if timestamp is not None:
        if not isinstance(timestamp, str):
            print('Timestamp must be a string')
            return result,None,None

    if units is not None:
        if not isinstance(units, str):
            print('units must be a string')
            return result,None,None
        if units != 'Jy' and units != 'Ta*' and units != 'Ta':
            print('units must be one of "Jy", "Ta*", or "Ta" - defaults to "Ta" if not specified')
            return result,None,None

    if not quiet:
        doTauWarning = tau is not None and units is None
        doApEffWarning = ap_eff is not None and units is None
        if doTauWarning and doApEffWarning:
            print('tau and ap_eff have been supplied but are not used by units="Ta"')
        else:
            if doTauWarning:
                print('tau has been supplied but is not used by units="Ta"')
            elif doApEffWarning:
                print('ap_eff has been supplied but is not used by the requested units')

    if bswitch is not None:
        if bswitch != 0 and bswitch != 1 and bswitch != 2:
            print('bswitch must be 0, 1, or 2')
            return result,None,None
        ret_bswitch = bswitch
    else:
        ret_bswitch = 0

    if sig_state is not None:
        if sig_state != 0 and sig_state != 1:
            print('sig_state must be 0 or 1')
            return result,None,None
        ret_sig_state = sig_state
    else:
        ret_sig_state = -1

    if checkref:
        if not refscan and not reftimestamp:
            print('The reference scan number is required unless a reftimestamp is provided.')
            return result,None,None

        # string argument type checks
        if reffile is not None:
            if not isinstance(reffile, str):
                print('Reffile must be a string')
                return result,None,None

        if reftimestamp is not None:
            if not isinstance(reftimestamp, str):
                print('Reftimestamp must be a string')
                return result,None,None

    # other checks and defaults
    retIfnum = 0
    retPlnum = 0
    retFdnum = 0
    retSampler = ''
    retInstance = 0
    retRefinstance = 0

    # indicate what defaults needs to be set
    # don't double check them if the user has explicitly set them
    checkIfnum = True
    checkPlnum = True
    checkFdnum = True

    # need the instance set appropriate first so that we can set the scan_info
    if instance is not None:
        if len(instance) > 1:
            print('Only one INSTANCE can be calibrated at a time')
            return result,None,None
        if instance[0] < 0:
            print('INSTANCE must be >= 0')
            return result,None,None
        retInstance = instance[0]

    # need the scan info so that we can set the defaults as necessary/appropriate
    info = find_scan_info(scan, timestamp=timestamp, instance=retInstance, filename=filename)
    infoOK = isinstance(info, tuple) and len(info) == 8

    nfd = 0
    nif = 0
    npl = 0

    if infoOK:
        iftabDim = info.iftable.ndim
        nif = iftabDim[0]
        nfd = 1 if len(iftabDim) < 2 else iftabDim[1]
        npl = 1 if len(iftabDim) < 3 else iftabDim[2]

    # checking default tuple in order: ifnum, fdnum, plnum
    # order matters - defaults when unset depend on previously set defaults in that order

    if ifnum is not None:
        if ifnum < 0:
            print('IFNUM must be >= 0')
            return result,None,None
        if infoOK and ifnum >= nif:
            print('IFNUM must be <', nif)
            return result,None,None
        retIfnum = ifnum
        checkIfnum = False

    if fdnum is not None:
        if fdnum < 0:
            print('FDNUM must be >= 0')
            return result,None,None
        if infoOK and fdnum >= nfd:
            print('FDNUM must be <', nfd)
            return result,None,None
        retFdnum = fdnum
        checkFdnum = False

    if plnum is not None:
        if plnum < 0:
            print('PLNUM must be >= 0')
            return result,None,None
        if infoOK and plnum >= npl:
            print('PLNUM must be <', npl)
            return result,None,None
        retPlnum = plnum
        checkPlnum = False

    if checkIfnum:
        # ifnum not set by user, set from info.iftable if possible
        if infoOK:
            count = 0
            ai = 0
            if checkFdnum:
                if checkPlnum:
                    # nothing already specified
                    loc = np.where(info.iftable)
                    if count > 0:
                        ai = getIftabIndices(loc, nif, nfd, npl)
                else:
                    # plnum specified
                    loc = np.where(info.iftable[:, :, retPlnum])
                    if count > 0:
                        ai = getIftabIndices(loc, nif, nfd, 1)
            else:
                if checkPlnum:
                    # fdnum specified, plnum is not
                    loc = np.where(info.iftable[:, retFdnum, :])
                    if count > 0:
                        ai = getIftabIndices(loc, nif, 1, npl)
                else:
                    # both fdnum and plnum are specified
                    loc = np.where(info.iftable[:, retFdnum, retPlnum])
                    if count > 0:
                        ai = getIftabIndices(loc, nif, 1, 1)
            if count > 0:
                # ai has dimensions [3,count] unless count=1
                # in either case, this form of indexing is OK
                # find whatever the minimum value is along the fdnum axis
                retIfnum = np.min(ai[0, :])

    if checkFdnum:
        # fdnum not set by user, retIfnum is now reliable either way
        # so use it
        if infoOK:
            count = 0
            ai = 0
            if checkPlnum:
                loc = np.where(info.iftable[retIfnum, :, :])
                if count > 0:
                    ai = getIftabIndices(loc, 1, nfd, npl)
            else:
                # plnum already specified
                loc = np.where(info.iftable[retIfnum, :, retPlnum])
                if count > 0:
                    ai = getIftabIndices(loc, 1, nfd, 1)
            if count > 0:
                retFdnum = np.min(ai[1, :])

    if checkPlnum:
        if infoOK:
            # only thing left to check, if count is positive then
            # the first found value is the appropriate plnum as is
            loc = np.where(info.iftable[retIfnum, retFdnum, :])
            count = len(loc[0])
            if count > 0:
                retPlnum = loc[0][0]

    if intnum is not None:
        if intnum < 0:
            print('INTNUM must be >= 0')
            return result,None,None

    if sampler is not None:
        if len(sampler) > 0:
            if fdnum is not None or ifnum is not None or plnum is not None:
                print('IFNUM, PLNUM, and FDNUM cannot be supplied when SAMPLER is supplied')
                return result,None,None
            retSampler = sampler
            if twofeeds is None:
                retFdnum = -1
            retIfnum = -1
            retPlnum = -1

    else:
        # ifnum, plnum, fdnum have been set either by the user or by the default finding mechanism.
        # Check that they are valid if info is valid
        if infoOK:
            if not info.iftable[retIfnum, retFdnum, retPlnum]:
                ifstr = 'IFNUM:' + str(retIfnum) + ' '
                fdstr = 'FDNUM:' + str(retFdnum) + ' '
                plstr = 'PLNUM:' + str(retPlnum) + ' '
                print('No data found at', ifstr, fdstr, plstr)
                showiftab(scan)
                return result,None,None

    if checkref:
        if refinstance is not None:
            if len(refinstance) > 1:
                print('Only one REFINSTANCE can be calibrated at a time')
                return result,None,None
            if refinstance[0] < 0:
                print('REFINSTANCE must be >= 0')
                return result,None,None
            retRefInstance = refinstance[0]
        # everything is okay
        ret = {'ifnum': retIfnum, 'plnum': retPlnum, 'fdnum': retFdnum, 'sampler': retSampler,
               'instance': retInstance, 'refinstance': retRefInstance, 'bswitch': ret_bswitch,
               'sig_state': ret_sig_state}
    else:
        # everything is okay
        ret = {'ifnum': retIfnum, 'plnum': retPlnum, 'fdnum': retFdnum, 'sampler': retSampler,
               'instance': retInstance, 'bswitch': ret_bswitch, 'sig_state': ret_sig_state}

    return 1,ret,info



def getfs(scan, ifnum=None, intnum=None, plnum=None, fdnum=None, sampler=None,
          tsys=None, tau=None, ap_eff=None, smthoff=None, units=None,
          nofold=None, blankinterp=None, nomask=None, eqweight=None,
          tcal=None, quiet=None, keepints=None, useflag=None, skipflag=None,
          instance=None, filename=None, timestamp=None, status=None):

    """
    This procedure retrieves and calibrates a frequency switched scan.  

    <p>This code could be used as a template for the user who may wish
    to develop more sophisticated calibration schemes. The spectrum is
    calibrated in Ta (K) by default.  Other recognized units are Ta* and
    Jy.

    <p><b>Summary</b>
    <ul><li>Data are selected using scan, ifnum, intnum, plnum and
    fdnum or, alternatively, sampler and intnum if you know the
    specific sampler name (e.g. "A10").
    
    <li>Individual integrations are processed separately.
    Each integration is processed using <a href="../toolbox/dofreqswitch.html">dofreqswitch</a>

    <li>The integrations are calibrated in Ta (K) by default.  If
    units of Ta* or Jy are requested via the units keyword, then 
    <a href="../toolbox/dcsetunits.html">dcsetunits</a> is used to convert to the desired units. 
    This produces two spectra, one where the "sig" phase of the
    integration was used as the "signal" and one where the "ref"
    phase of the integration was used as the "signal".

    <li>The two resulting data containers are combined using 
    <a href="../toolbox/dcfold.html">dcfold</a> unless the nofold keyword is set.  This step is also 
    skipped if the there is no frequency overlap between the two
    spectra (the frequency switching distance is more than the
    total number of channels).  In that case (out-of-band
    frequency switching), the data can not be "fold"ed and this
    step is skipped. 
    
    <li>Averaging of individual integrations is then done using 
    <a href="../toolbox/dcaccum.html">dcaccum</a>.  By default, integrations are weighted as described in dcaccum.  
    If the eqweight keyword is set, then integrations are averaged with an
    equal weight.  If the nofold keyword is set or the data is
    out-of-band frequency-switched then the two results  are
    averaged separately for each integration.   

    <li>The final average is left in the primary data container
    (buffer 0), and a summary line is printed. If the nofold keyword
    is set then the other result is left in buffer 1.  These results
    can be combined later by the user using <a href="fold.html">fold</a>  The printing of the
    summary line can be suppressed by setting the quiet keyword. 

    <li>The individual integration results can be saved to the
    currently opened output file by setting the keepints keyword.  The
    final average is still produced in that case.  If the nofold
    keyword is set, the "signal" result is kept first followed by the
    "reference" result for each integration, otherwise only the
    folded result is saved for each integration.  In the case of
    out-of-band frequency-switched data only the "signal" result is
    saved unless the nofold keyword is explicitly set.
    </ul>
    <p><b>Parameters</b>
    <p>
    The only required parameter is the scan number.  Arguments to
    identify the IF number, polarization number, and feed number are
    optional. 
    <p>
    <p> If ifnum, fdnum, or plnum are not supplied then the lowest
    values for each of those where data exists (all combinations may not
    have data) will be used, using any user-supplied values.  The value
    of ifnum is determined first, followed by fdnum and finally plnum.  If a
    combination with data can not be found then <a href="showiftab.html">showiftab</a>
    is used to show the user what the set of valid combinations are.
    The summary line includes the ifnum, fdnum, and plnum used.
    <p>
    <b>Tsys and Available Units</b>
    <p>
    The procedure calculates Tsys based on the Tcal values
    and the data.  The user can override this calculation by 
    entering a zenith system temperature.  The procedure will then correct the 
    user-supplied Tsys for the observed elevation.  If the data are
    calibrated to Ta* or Jy,  additional parameters are used.  A zenith
    opacity (tau) may be specified, and an aperture efficiency may be
    specified.  The user is strongly encouraged to enter values for
    these calibration parameters, but they will be estimated if none are
    provided.  The user can also supply a mean tcal using the tcal
    keyword.  That will override the tcal found in the data.
    <p>
    <b>Smoothing the Reference Spectra</b>
    <p>
    A parameter called smthoff can be used to smooth the reference
    spectrum in dofreqswitch.  In certain cases this can improve the
    signal to noise ratio, but it may degrade baseline shapes and
    artificially emphasize spectrometer glitches.  Use with care.  A
    value of smthoff=16 is often a good choice. 
    <p> 
    <b>Weighting of Integrations in Scan Average</b>
    <p> 
    By default, the averaging of integrations is weighted using tsys,
    exposure, and frequency_resolution as described in the <a href="../toolbox/dcaccum.html">dcaccum</a> 
    documentation. To give all integrations equal weight instead of the
    default weighting based on Tsys, use the /eqweight keyword.
    <p>
    If the data were taken with out-of-band frequency switching then no folding 
    will be done and the nofold argument is ignored.
    <p>
    <b>Using or Ignoring Flags</b>
    <p>
    Flags (set via <a href="flag.html">flag</a>) can be selectively
    applied or ignored using the useflag and skipflag keywords.  Only one of
    those two keywords can be used at a time (it is an error to use both
    at the same time).  Both can be either a boolean (/useflag or /skipflag)
    or an array of strings.  The default is /useflag, meaning that all flag
    rules that have been previously set are applied when the data is
    fetched from disk, blanking data as described by each rule.  If
    /skipflag is set, then all of the flag rules associated with this data
    are ignored and no data will be blanked when fetched from disk (it
    may still contain blanked values if the actual values in the disk
    file have already been blanked by some other process).  If useflag is a
    string or array of strings, then only those flag rules having the
    same idstring value are used to blank the data.  If skipflag is a
    string or array of strings, then all flag rules except those
    with the same idstring value are used to blank the data.
    <p>
    <b>Dealing with flagged channels</b>
    <p>
    When individual channels are flagged in the raw data (e.g. VEGAS
    spikes at the expected spike locations) the data values at those
    channels are replaced with Not a Number when the data is read from
    disk.  That presents a challenge when processing frequency switched
    data to avoid a spike appearing at the flagged channel locations
    after the fold step done by this procedure (unless nofold is
    selected).  When the data are combined at the fold step, each
    channel data average is the weighted average (using Tsys) of two
    data values, each from the same sky frequency but from different
    original channels in the raw data.  When indivual channels are
    flagged, that average can consist of just one finite value because
    the frequency shift will seldom lead to overlapping flagged
    channels. If there is any significant baseline structure across the
    bandpass then that single finite channel will be noticeably
    different from the local average of two channels.  That will lead to
    a noticable spike (positive or negative) at the location of a
    flagged channel, which is exactly the behavior that flagging the
    channel was trying to avoid.  

    <p>
    This procedure offers two ways of dealing with flagged channels to
    avoid that problem.  The default makes sure that both the original
    flagged channel and the corresponding channel in the other spectrum,
    after one of them has been shifted to align in frequency, is also
    flagged so that the final average has no finite values for that
    channel (i.e. it appears as a flagged channel).  Alternatively, the
    blankinterp keyword can be used to tell the fold procedure to
    interpolate across all blanked values before doing any shifting and
    averaging.  In the case of individually flagged channels, the
    blanked channel is replaced by the average of the two adjacent
    channels.  This obviously adds a new data value at the previously
    unknown (flagged) channel but it can make downstream data processing
    simpler by not having to worry that some of the channels contain
    non-finite values.  That may be important if the data are exported
    out of GBTIDL.  Finally, the nomask keyword can be used to turn
    off this special handling (masking) of flagged channels before the
    average (where spikes may result).  If blankinterp is used then
    nomask has no effect because the data are interpolated before the
    masking step happens.  If nofold is used then the data are never
    masked or interpolated.
    <p>
    <b>Dealing With Duplicate Scan Numbers</b>
    <p>
    There are 3 ways to attempt to resolve ambiguities when the
    same scan number appears in the data source.  The instance keyword
    refers to the element of the returned array of scan_info structures
    that <a href="scan_info.html">scan_info</a> returns.  So, if scan 23
    appears 3 times then instance=1 refers to the second time that scan 23
    appears as returned by scan_info.  The filename keyword is useful if a 
    scan is unique to a specific file and multiple files have been accessed
    using <a href="dirin.html">dirin</a>.  If filename is specified and instance
    is also specified, then instance refers to the instance of that scan
    just within that file (which may be different from its instance within
    all opened files when dirin is used).  The timestamp keyword is another
    way to resolve ambiguous scan numbers.  The timestamp here is a string
    used essentially as a label by the monitor and control system and is
    unique to each scan.  The format of the timestamp string is
    "YYYY_MM_DD_HH:MM:SS".  When timstamp is given, scan and instance
    are ignored.  If more than one match is found, an error is 
    printed and this procedure will not continue.  

    @param scan {in}{required}{type=integer} scan number
    @keyword ifnum {in}{optional}{type=integer} IF number
    (starting with 0). Defaults to the lowest value associated with data
    taking into account any user-supplied values for fdnum, and plnum.
    @keyword intnum {in}{optional}{type=integer} integration number,
    default is all integrations.
    @keyword plnum {in}{optional}{type=integer} Polarization number
    (starting with 0).  Defaults to the lowest value with data after
    determining the values of ifnum and fdnum if not supplied by the
    user.
    @keyword fdnum {in}{optional}{type=integer} Feed number.  Defaults
    to the lowest value with data after determining the value of ifnum
    if not supplied by the user and using any value of plnum supplied by
    the user.  
    @keyword sampler {in}{optional}{type=string} sampler name, this is
    an alternative way to specify ifnum,plnum, and fdnum.  When sampler
    name is given, ifnum, plnum, and fdnum must not be given.
    @keyword tsys {in}{optional}{type=float} tsys at zenith, this is
    converted to a tsys at the observed elevation.  If not suppled, the
    tsys for each integration is calculated as described elsewhere.
    @keyword tau {in}{optional}{type=float} tau at zenith, if not
    supplied, it is estimated using <a href="../toolbox/get_tau.html">get_tau</a>
    tau is only used when the requested units are other than the default
    of Ta and when a user-supplied tsys value at zenith is to be used.
    @keyword ap_eff {in}{optional}{type=float} aperture efficiency, if
    not suppled, it is estimated using <a href="../toolbox/get_ap_eff.html">get_ap_eff<a>
    ap_eff is only used when the requested units are Jy.
    @keyword smthoff {in}{optional}{type=integer} smooth factor for reference spectrum
    @keyword units {in}{optional}{type=string} takes the value 'Jy',
    'Ta', or 'Ta*', default is Ta.
    @keyword nofold {in}{optional}{type=boolean} When set, getfs does not fold
    the calibrated spectrum.  Buffer 0 then contains the result of
    (sig-ref)/ref while buffer 1 contains the result of
    (ref-sig)/sig, calibrated independently and averaged over all
    integrations.  Only data container 0 will be shown on the plotter.
    Default is unset (folded result).
    @keyword blankinterp {in}{optional}{type=boolean} When set, blanks
    are replaced, before the fold step, by a linear interpolation
    using the finite values found in the two spectra.  For isolated blanked
    channels, the replacement value is the average of the two adjacent
    channel values.  This argument is ignored if nofold is used.
    @keyword nomask {in}{optional}{type=boolean} When set, turn off the
    masking of blank channels from each spectrum on to the other, after
    the shift, when folding the data.  This may result in spikes at the
    location of blanked channels. This was the original behavior of this
    routine. This keyword is ignored if /nofold is used.
    @keyword eqweight {in}{optional}{type=boolean} When set, all
    integrations are averaged with equal weight (1.0).  Default is unset.
    @keyword tcal {in}{optional}{type=float} Cal temperature (K) to use
    in the Tsys calculation.  If not supplied, the mean_tcal value from
    the header of the cal_off switching phase data in each integration
    is used.  This must be a scalar, vector tcal is not yet supported.
    The resulting data container(s) will have it's mean_tcal header value
    set to this keyword when it is set by the user.
    @keyword quiet {in}{optional}{type=boolean} When set, the normal
    status message on successful completion is not printed.  This keyword will
    not affect error messages.  Default is unset.
    @keyword keepints {in}{optional}{type=boolean} When set, the
    individual integrations are saved to the current output file
    (as set by fileout).  This keyword is ignored if an integration is
    specified using the intnum keyword.  Default is unset.
    @keyword useflag {in}{optional}{type=boolean or string}
    Apply all or just some of the flag rules?  Default is set.
    @keyword skipflag {in}{optional}{type=boolean or string} Do not apply
    any or do not apply a few of the flag rules?  Default is unset.
    @keyword instance {in}{optional}{type=integer} Which occurence
    of this scan should be used.  Default is 0.
    @keyword filename {in}{optional}{type=string} When specified, limit the search 
    for this scan (and instance) to this specific file.  Default is all files.
    @keyword timestamp {in}{optional}{type=string} The M&C timestamp associated
    with the desired scan. When supplied, scan and instance are ignored.
    @keyword status {out}{optional}{type=integer} An output value to indicate
    whether the procedure finished as expected.  A value of 1 means there were
    no problems, a value of -1 means there were problems with the
    arguments before any data was processed, and a value of 0 means that
    some of the individual integrations were processed (and possibly
    saved to the output file if keepints was set) but there was a
    problem with the final average, and the contents of the PDC remain
    unchanged. 
    
    @examples
    Typical use of getfs:
    <pre>
    getfs,76
    accum
    getfs,77
    accum
    ave
    show
    </pre>
 

    In the following example, the spectrum is not folded and the two components
    of the calibration are shown overlaid on the plotter.  Then the data are
    folded 'by hand'. This example also shows how /skipflag can be used to
    ignore all previously set flags.

    <pre>
    getfs,76,/nofold,/skipflag
    oshow,1
    fold
    </pre>

    @uses <a href="../toolbox/accumave.html">accumave</a>
    @uses <a href="../toolbox/accumclear.html">accumclear</a>
    @uses <a href="../../devel/guide/calsummary.html">calsummary</a>
    @uses <a href="../../devel/guide/check_calib_args.html">check_calib_args</a>
    @uses <a href="../toolbox/data_free.html">data_free</a>
    @uses <a href="../toolbox/dcaccum.html">dcaccum</a>
    @uses <a href="../toolbox/dcfold.html">dcfold</a>
    @uses <a href="../toolbox/dcscale.html">dcscale</a>
    @uses <a href="../toolbox/dcsetunits.html">dcsetunits</a>
    @uses <a href="../toolbox/dofreqswitch.html">dofreqswitch</a>
    @uses <a href="../../devel/guide/find_scan_info.html">find_scan_info</a>
    @uses <a href="../../devel/guide/get_calib_data.html">get_calib_data</a>
    @uses <a href="set_data_container.html">set_data_container</a>
    @uses <a href="showiftab.html">showiftab</a>
    
    """
    
    status = -1

    # basic argument checks
    argsOK,ret,info = check_calib_args(scan, ifnum=ifnum, intnum=intnum, plnum=plnum,
                                       fdnum=fdnum, sampler=sampler, eqweight=eqweight,
                                       units=units, quiet=quiet, keepints=keepints,
                                       useflag=useflag, skipflag=skipflag, instance=instance,
                                       filename=filename, timestamp=timestamp, tau=tau,
                                       ap_eff=ap_eff)
    if not argsOK:
        return

    if info is None:
        return

    # FS data must have 4 swiching states, 2 CAL states and 2 SIG states.
    if info.n_switching_states != 4:
        print('This does not appear to be standard frequency switched data.')
        if info.n_cal_states != 2:
            print('The number of cal states is not 2, as needed for this procedure.')
        if info.n_sig_states != 2:
            print('The number of sig states is not 2, as needed for this procedure.')
        return

    # Get the requested data
    data = get_calib_data(info, ret['ifnum'], ret['plnum'], ret['fdnum'],
                          ret['sampler'], count, intnum=intnum,
                          useflag=useflag, skipflag=skipflag)

    if count <= 0:
        print("No data found, can not continue.")
        return

    # from this point on, data contains data containers that must be freed
    # whenever this routine returns to avoid memory leaks

    # this test isn't available via scan_info, it has to wait until now
    if data[0].switch_state != 'FSWITCH':
        print('This is apparently not frequency switched data. switch_state = ' + data[0].switch_state)
        return

    # find the 4 types of data container
    # signal with cal, signal without cal,
    # reference with cal, reference without cal
    sigwcal = np.where((data.cal_state == 1) & (data.sig_state == 1))[0]
    sig = np.where((data.cal_state == 0) & (data.sig_state == 1))[0]
    refwcal = np.where((data.cal_state == 1) & (data.sig_state == 0))[0]
    ref = np.where((data.cal_state == 0) & (data.sig_state == 0))[0]

    # Final sanity checks

    # In this calibration, we calibrate each integration separately
    # and then average the results. That means that we need the same
    # number of data containers of each type, one per integration.
    if intnum is None:
        expectedCount = 1
    else:
        # find appropriate number for this sampler
        sampIndx = np.where(info.samplers == data[0].sampler_name)[0][0]
        expectedCount = info.n_sampints[sampIndx]

    if (countSigwcal != expectedCount or countSigwcal != countSig or countSigwcal != countRefwcal or
            countSigwcal != countRef):
        print("Unexpected number of spectra retrieved for some or all of the switching phases, can not continue.")
        tb.data_free(data)
        return

    # watch for out-of-band frequency switching, it's okay, just turn on nofold if true.
    thisnofold = nofold is not None
    sig0 = sig[0]
    ref0 = ref[0]
    sigF0 = chantofreq(data[sig0], 0.0)
    refF0 = chantofreq(data[ref0], 0.0)
    chan_shift = (refF0 - sigF0) / data[sig0].frequency_interval
    npts = data_valid(data[sig0])
    if abs(chan_shift) >= npts:
        thisnofold = True

    status = 0
    missing = 0

    weight = 1.0 if eqweight is not None else None

    res1accum = AccumStruct()
    if thisnofold:
        res2accum = AccumStruct()
    tauInts = np.zeros(expectedCount)
    apEffInts = np.zeros(expectedCount)
    for n_int in range(expectedCount):
        tb.dofreqswitch(data[sigwcal[n_int]], data[sig[n_int]], data[refwcal[n_int]],
                        data[ref[n_int]], smthoff,tsys=tsys, tau=tau, tcal=tcal,
                        sigResult=sigResult, refResult=refResult)

        if thisnofold:
            # convert units on both result
            tb.dcsetunits(sigResult, units, tau=tau, ap_eff=ap_eff)
            tb.dcsetunits(refResult, units, tau=tau, ap_eff=ap_eff,
                          ret_tau=ret_tau, ret_ap_eff=ret_ap_eff)
        else:
            # fold the two results
            folded = tb.dcfold(sigResult, refResult, blankinterp=blankinterp,
                               nomask=nomask)
            tb.data_copy(folded, sigResult)
            tb.data_free(folded)
            # and convert the units
            tb.dcsetunits(sigResult, units, tau=tau, ap_eff=ap_eff,
                               ret_tau=ret_tau, ret_ap_eff=ret_ap_eff)

        # these are only used in the status line at the end
        tauInts[n_int] = ret_tau
        apEffInts[n_int] = ret_ap_eff

        tb.dcaccum(res1accum, sigResult, weight=weight)
        if thisnofold:
            tb.dcaccum(res2accum, refResult, weight=weight)
        if keepints:
            # re-use existing DCs so space isn't wasted.
            # can't use the array element directly as it would
            # not be passed by reference so would not be changed
            # won't work: data_copy(sigResult, data[sig[n_int]])
            # instead, do this
            tmp = data[sig[n_int]]  # gets the right pointer
            tb.data_copy(sigResult, tmp)  # copies header, re-uses pointer
            data[sig[n_int]] = tmp  # puts it back in place
            if nofold:
                # same here
                tmp = data[ref[n_int]]  # gets the right pointer
                tb.data_copy(refResult, tmp)  # copies header, re-uses pointer
                data[ref[n_int]] = tmp  # puts it back in place

    if keepints:
        putchunk(data[sig])
        if nofold:
            putchunk(data[ref])

    naccum1 = res1accum.n
    if naccum1 <= 0:
        print('Result is all blanked - probably all of the data were flagged')
        # sigResult must therefore be all blanked, use it as the end result
        sigResult = sigResult
        tb.data_free(sigResult)
        if tb.data_valid(refResult) > 0:
            tb.data_free(refResult)
        if tb.data_valid(res2accum) > 0:
            tb.data_free(res2accum)
        tb.data_free(data)
        return

    accumave(res1accum, sigResult, quiet=True)
    missing = naccum1 != expectedCount
    if thisnofold:
        naccum2 = res2accum.n
        if naccum2 <= 0:
            print('Result in buffer 1 is all blanked - probably at least one phase in each integration was blanked.')
            print('Can not continue - units of result in primary data container may not be as expected.')
            # refResult must be all blanked, use it as the result in buffer 1
            refResult = refResult
            # clean up
            tb.data_free(sigResult)
            tb.data_free(refResult)
            tb.data_free(data)
            return
        tb.accumave(res2accum, refResult, quiet=True)
        missing = missing or naccum2 != expectedCount

    status = 1
    tb.set_data_container(sigResult)
    if thisnofold:
        tb.set_data_container(refResult, buffer=1)
    if not quiet:
        if missing:
            nmiss = expectedCount - naccum1
        tb.calsummary(info.scan, sigResult.tsys, sigResult.units, tauInts=tauInts,
                      apEffInts=apEffInts, missingInts=nmiss, ifnum=ret['ifnum'],
                      plnum=ret['plnum'], fdnum=ret['fdnum'])

    tb.data_free(data)
    tb.data_free(sigResult)
    if tb.data_valid(refResult) > 0:
        tb.data_free(refResult)
