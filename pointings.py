import numpy as np
from qubic import QubicSampling, equ2hor
from astropy.time import TimeDelta
from pdb import set_trace
from pyoperators import (
    Cartesian2SphericalOperator, Rotation3dOperator,
    Spherical2CartesianOperator)
from pysimulators import CartesianEquatorial2HorizontalOperator, SphericalHorizontal2EquatorialOperator
from copy import deepcopy

def create_sweeping_pointings(parameter_to_change= None,
                              value_of_parameter = None,
                              center=[0.0, -46.],
                              duration=24,
                              sampling_period=0.05,
                              angspeed=1,
                              delta_az=30,
                              nsweeps_per_elevation=320,
                              angspeed_psi=1,
                              maxpsi=15,
                              date_obs=None,
                              latitude=None,
                              longitude=None,
                              return_hor=True,
                              delta_nsw=0.0,
                              ss_az='sss',
                              ss_el=None,
                              ss_psi='sss',
                              hwp_div=8,
                              dead_time=5 # sec
                              ):
    """
    Return pointings according to the sweeping strategy:
    Sweep around the tracked FOV center azimuth at a fixed elevation, and
    update elevation towards the FOV center at discrete times.

    Parameters
    ----------
    center : array-like of size 2
        The R.A. and Declination of the center of the FOV.
    duration : float
        The duration of the observation, in hours.
    sampling_period : float
        The sampling period of the pointings, in seconds.
    angspeed : float
        The pointing angular speed, in deg / s.
    delta_az : float
        The sweeping extent in degrees.
    nsweeps_per_elevation : int
        The number of sweeps during a phase of constant elevation.
    angspeed_psi : float
        The pitch angular speed, in deg / s.
    maxpsi : float
        The maximum pitch angle, in degrees.
    latitude : float, optional
        The observer's latitude [degrees]. Default is DOMEC's.
    longitude : float, optional
        The observer's longitude [degrees]. Default is DOMEC's.
    date_obs : str or astropy.time.Time, optional
        The starting date of the observation (UTC).
    return_hor : bool, optional
        Obsolete keyword.
    hwp_div : number of positions in rotation of hwp between 0. and 90. degrees

    Returns
    -------
    pointings : QubicPointing
        Structured array containing the azimuth, elevation and pitch angles,
        in degrees.

    """
    do_fast_sweeping = False
    standard_scanning_az = False
    standard_scanning_el = True
    if ss_az == None or ss_az == 'sss' or ss_az[:-4] == 'sss':
        standard_scanning_az = True
    if ss_az != None and ss_az[-4:] == '_fsw':
        do_fast_sweeping = True

    if parameter_to_change != None and value_of_parameter != None:
        exec parameter_to_change + '=' + str(value_of_parameter)
        
    nsamples = int(np.ceil(duration * 3600 / sampling_period))
    out = QubicSampling(
        nsamples, date_obs=date_obs, period=sampling_period,
        latitude=latitude, longitude=longitude)
    racenter = center[0]
    deccenter = center[1]
    backforthdt = delta_az / angspeed * 2 + 2 * dead_time

    # compute the sweep number
    isweeps = np.floor(out.time / backforthdt).astype(int)

    # azimuth/elevation of the center of the field as a function of time
    azcenter, elcenter = equ2hor(racenter, deccenter, out.time,
                                 date_obs=out.date_obs, latitude=out.latitude,
                                 longitude=out.longitude)

    # compute azimuth offset for all time samples
    daz = out.time * angspeed
    daz = daz % ((delta_az + dead_time / angspeed) * 2)
    mask = daz > (delta_az + dead_time / angspeed)
    daz[mask] = -daz[mask] + 2 * delta_az + dead_time / angspeed
    daz -= delta_az / 2
    
    if do_fast_sweeping:
        delta_az_fs = 2. #deg
        B = 3*np.pi*angspeed / (2*delta_az_fs)
        fast_sweeping = delta_az_fs * np.sin(B*out.time)
        daz += fast_sweeping

    # elevation is kept constant during nsweeps_per_elevation
    elcst = np.zeros(nsamples)
    ielevations = isweeps // nsweeps_per_elevation
    nsamples_per_sweep = int(backforthdt / sampling_period)
    iel = 0

    nelevations = ielevations[-1] + 1
    for i in xrange(nelevations):
        mask = ielevations == i
        elcst[mask] = np.mean(elcenter[mask])
                
    # azimuth and elevations to use for pointing
    azptg = azcenter + daz
    if standard_scanning_el:
        elptg = elcst
            
    ### scan psi as well
    if maxpsi == 0:
        pitch = np.zeros(len(out.time))
    elif ss_psi == 'sss':
        pitch = out.time * angspeed_psi
        pitch = pitch % (4 * maxpsi)
        mask = pitch > (2 * maxpsi)
        pitch[mask] = -pitch[mask] + 4 * maxpsi
        pitch -= maxpsi
    elif ss_psi == 'const_during_hwp_full_rot':
        pitch = np.ones(nsamples) * maxpsi
        mask = (out.time // (backforthdt * hwp_div / 2)) % 2 == 1
        pitch[mask] *= -1

    # HWP rotating
    hwp = np.floor(out.time*2 / backforthdt).astype(float)
    hwp  = hwp % (hwp_div * 2)
    hwp[hwp > hwp_div] = -hwp[hwp > hwp_div] + hwp_div*2
    hwp *= 90. / hwp_div

    out.azimuth = azptg
    out.elevation = elptg
    out.pitch = pitch
    out.angle_hwp = hwp
    
    # mask pointings during the dead time
    mask = (daz < delta_az / 2) * (daz > - delta_az / 2)
    out = out[mask]
    
    return out

def corrupt_pointing(pointing,
                     sigma_azimuth=0, # arcmin
                     sigma_elevation=0,
                     sigma_psi=0,
                     units='arcmin',
                     seed=0):
    #p = deepcopy(pointing)
    p = QubicSampling(azimuth=pointing.azimuth.copy(), 
                      elevation=pointing.elevation.copy(),
                      pitch=pointing.pitch.copy(),
                      period=pointing.period,
                      latitude=pointing.latitude,
                      longitude=pointing.longitude)
    p.angle_hwp = pointing.angle_hwp.copy()
    p.date_obs = pointing.date_obs.copy()
    np.random.seed(seed)
    nsamples = len(p)
    if units not in ('arcsec', 'arcmin', 'deg'):
        raise ValueError('Wrong units for corrupt_pointing function')
    coef = {'arcsec': 3600,
            'arcmin': 60,
            'deg': 1}[units]
    if sigma_azimuth != 0:
        p.azimuth += np.random.normal(0, sigma_azimuth/coef, nsamples)
    if sigma_elevation != 0:
        p.elevation += np.random.normal(0, sigma_elevation/coef, nsamples)
    if sigma_psi != 0:
        p.pitch += np.random.normal(0, sigma_psi/coef, nsamples)
    return p

