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
                              delta_el_corr=0.0,
                              ss_az='sss',
                              ss_el='el_enlarged1',
                              hwp_div=8,
                              dead_time=0):
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
    sin_az_changing = False
    do_fast_sweeping = False
    standard_scanning_az = False
    standard_scanning_el = False
    progressive1_sin_az_ch = False
    progressive2_sin_az_ch = False
    no_ss_az = False
    no_ss_el = False
    el_enlarged1 = False
    el_enlarged2 = False
    progr_el = False
    if ss_az == None or ss_az == 'sss' or ss_az[:-4] == 'sss':
        standard_scanning_az = True
    elif ss_az == 'sin_az' or ss_az[:-4] == 'sin_az':
        sin_az_changing = True
    elif ss_az == 'progr1_sin_az' or ss_az[:-4] == 'progr1_sin_az':
        progressive1_sin_az_ch = True
    elif ss_az == 'progr2_sin_az' or ss_az[:-4] == 'progr2_sin_az':
        progressive2_sin_az_ch = True
    elif ss_az == 'lin_az' or ss_az[:-4] == 'lin_az':
        pass
    elif ss_az == 'no_ss':
        no_ss_az = True
    if ss_az != None and ss_az[-4:] == '_fsw':
        do_fast_sweeping = True

    if ss_el == 'progr_el':
        progr_el = True
    elif ss_el == 'el_enlarged1':
        el_enlarged1 = True
    elif ss_el == 'el_enlarged2':
        el_enlarged2 = True
    elif ss_el == 'no_ss':
        no_ss_el = True
    else:
        standard_scanning_el = True

    if parameter_to_change != None and value_of_parameter != None:
        exec parameter_to_change + '=' + str(value_of_parameter)

    nsamples = int(np.ceil(duration * 3600 / sampling_period))
    out = QubicSampling(
        nsamples, date_obs=date_obs, period=sampling_period,
        latitude=latitude, longitude=longitude)
    racenter = center[0]
    deccenter = center[1]
    backforthdt = delta_az / angspeed * 2

    # compute the sweep number
    isweeps = np.floor(out.time / backforthdt).astype(int)

    # azimuth/elevation of the center of the field as a function of time
    azcenter, elcenter = equ2hor(racenter, deccenter, out.time,
                                 date_obs=out.date_obs, latitude=out.latitude,
                                 longitude=out.longitude)

    # compute azimuth offset for all time samples
    if sin_az_changing:
        daz = - delta_az/2. * np.cos(out.time * 2*np.pi / backforthdt)
    elif progressive1_sin_az_ch:
        daz = - delta_az/2. * np.cos(out.time * 2*np.pi / backforthdt)
        A = 0.04
        daz *= A*(1-np.cos(4*out.time * 2*np.pi / backforthdt)) + 1
    elif progressive2_sin_az_ch:
        daz = out.time * angspeed
        daz = daz % (delta_az * 2)
        mask = daz > delta_az
        daz[mask] = -daz[mask] + 2 * delta_az
        daz -= delta_az / 2
        sin_daz = - delta_az/2. * np.cos(out.time * 2*np.pi / backforthdt)
        daz = 2*daz - sin_daz
    elif no_ss_az:
        daz = 0.
    else:
        daz = out.time * angspeed
        daz = daz % (delta_az * 2)
        mask = daz > delta_az
        daz[mask] = -daz[mask] + 2 * delta_az
        daz -= delta_az / 2
    
    if do_fast_sweeping:
        delta_az_fs = 2. #deg
        B = 3*np.pi*angspeed / (2*delta_az_fs)
        fast_sweeping = delta_az_fs * np.sin(B*out.time)
        daz += fast_sweeping

    # correction on RA and declination
    x = (elcenter - np.min(elcenter)) * np.pi / (np.max(elcenter) - np.min(elcenter))
    elevation_corr = np.zeros(nsamples)
    nsweeps_corr = np.zeros(nsamples)
    mask = (x < np.pi/4.) + (x > np.pi*3./4.)
    elevation_corr[mask] = np.sin(8*x[mask]) * delta_az*delta_el_corr
    nsweeps_corr[mask] = nsweeps_per_elevation * (1-delta_nsw)
    mask = np.invert(mask)
    elevation_corr[mask] = 0.
    nsweeps_corr[mask] = nsweeps_per_elevation
    
    # elevation is kept constant during nsweeps_per_elevation
    elcst = np.zeros(nsamples)
    elcst_m = np.zeros(nsamples)
    ielevations = isweeps // nsweeps_per_elevation
    ielevations_m = np.array([]).astype(int)
    nsamples_per_sweep = int(backforthdt / sampling_period)
    iel = 0
    while len(ielevations_m) < nsamples:
        add = (np.ones(nsweeps_corr[len(ielevations_m)]*nsamples_per_sweep)*iel).astype(int)
        if iel == 0:
            ielevations_m = add
        else:
            ielevations_m = np.concatenate((ielevations_m, add), axis=1)
        iel += 1
    ielevations_m = ielevations_m[:nsamples]

    nelevations = ielevations[-1] + 1
    nelevations_m = ielevations_m[-1] + 1
    if el_enlarged1:
        min_el = np.min(elcenter)
        elcenter -= min_el
        elcenter *= 1. - delta_el_corr
        elcenter += min_el
    elif el_enlarged2:
        elcenter = np.ones(nsamples) * np.min(elcenter)
    for i in xrange(nelevations):
        mask = ielevations == i
        elcst[mask] = np.mean(elcenter[mask])
    for i in xrange(nelevations_m):
        mask = ielevations_m == i
        elcst_m[mask] = np.mean(elcenter[mask]) + np.mean(elevation_corr[mask])
                
    # azimuth and elevations to use for pointing
    azptg = azcenter + daz
    if standard_scanning_el or el_enlarged1 or el_enlarged2:
        elptg = elcst
    elif no_ss_el:
        elptg = elcenter
    elif progr_el:
        elptg = elcst_m
            
    ### scan psi as well
    if maxpsi == 0 or angspeed_psi == 0:
        pitch = np.zeros(len(out.time))
    else:
        pitch = out.time * angspeed_psi
        pitch = pitch % (4 * maxpsi)
        mask = pitch > (2 * maxpsi)
        pitch[mask] = -pitch[mask] + 4 * maxpsi
        pitch -= maxpsi

    # HWP rotating
    hwp = np.floor(out.time*2 / backforthdt).astype(float)
    hwp  = hwp % (hwp_div * 2)
    hwp[hwp > hwp_div] = -hwp[hwp > hwp_div] + hwp_div*2
    hwp *= 90. / hwp_div

    out.azimuth = azptg
    out.elevation = elptg
    out.pitch = pitch
    out.angle_hwp = hwp
    
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
