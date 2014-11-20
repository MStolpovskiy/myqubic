import numpy as np
from qubic.pointings import *
from astropy.time import TimeDelta
from pdb import set_trace
from pyoperators import (
    Cartesian2SphericalOperator, Rotation3dOperator,
    Spherical2CartesianOperator)
from pysimulators import CartesianEquatorial2HorizontalOperator, SphericalHorizontal2EquatorialOperator

def create_sweeping_pointings_many_fields(
        center, duration, sampling_period, angspeed, delta_az,
        nsweeps_per_elevation, angspeed_psi, maxpsi, date_obs=None,
        latitude=None, longitude=None, return_hor=True, 
        delta_el_corr=0.1, file_to_save=None):
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

    Returns
    -------
    pointings : QubicPointing
        Structured array containing the azimuth, elevation and pitch angles,
        in degrees.

    """
    nsamples = int(np.ceil(duration * 3600 / sampling_period))
    out = QubicPointing.zeros(
        nsamples, date_obs=date_obs, sampling_period=sampling_period,
        latitude=latitude, longitude=longitude)
    racenter = center.T[0]
    deccenter = center.T[1]
    backforthdt = delta_az / angspeed * 2

    days = (np.arange(nsamples) // (24 * 3600 / sampling_period))
    ## for iday in xrange(int(np.max(days))):
    ##     mask_day = days == iday
    
    # compute the sweep number
    isweeps = np.floor(out.time / backforthdt).astype(int)

    # azimuth/elevation of the center of the field as a function of time
    azcenter = np.empty(nsamples)
    elcenter = np.empty(nsamples)
    for iday in xrange(int(np.max(days)) + 1):
        mask = days == iday
        i = iday - (iday // 3) * 3
        azcenter[mask], elcenter[mask] = equ2hor(racenter[i],
                                                 deccenter[i],
                                                 out.time[mask],
                                                 date_obs=out.date_obs,
                                                 latitude=out.latitude,
                                                 longitude=out.longitude)
    
    # compute azimuth offset for all time samples
    daz = - delta_az/2. * np.cos(out.time * 2*np.pi / backforthdt)
    A = 0.04
    daz *= A*(1-np.cos(4*out.time * 2*np.pi / backforthdt)) + 1
    
    if False: # Fast Sweeping (FSW)
        delta_az_fs = 2. #deg
        B = 3*np.pi*angspeed / (2*delta_az_fs)
        fast_sweeping = delta_az_fs * np.sin(B*out.time)
        daz += fast_sweeping

    # elevation is kept constant during nsweeps_per_elevation
    elcst = np.zeros(nsamples)
    ielevations = isweeps // nsweeps_per_elevation
    nsamples_per_sweep = int(backforthdt / sampling_period)
    nelevations = ielevations[-1] + 1

    for iday in xrange(int(np.max(days)) + 1):
        mask = days == iday
        min_el = np.min(elcenter[mask])
        elcenter[mask] -= min_el
        elcenter[mask] *= 1. - delta_el_corr
        elcenter[mask] += min_el

    for i in xrange(nelevations):
        mask = ielevations == i
        elcst[mask] = np.mean(elcenter[mask])
                
    # azimuth and elevations to use for pointing
    azptg = azcenter + daz
    elptg = elcst
            
    ### scan psi as well
    pitch = out.time * angspeed_psi
    pitch = pitch % (4 * maxpsi)
    mask = pitch > (2 * maxpsi)
    pitch[mask] = -pitch[mask] + 4 * maxpsi
    pitch -= maxpsi

    out.azimuth = azptg
    out.elevation = elptg
    out.pitch = pitch

    return out

