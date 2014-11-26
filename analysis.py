from __future__ import division
import numpy as np
import healpy as hp
from itertools import izip
import random
from qubic import (
    map2tod, tod2map_all, tod2map_each, QubicScene, QubicAcquisition, QubicInstrument)
from pdb import set_trace
from copy import copy

class QubicAnalysis(object):
    def __init__(self,
                 acquisition_or_rec_map,
                 input_map,
                 coverage=None,
                 mode='all',
                 coverage_thr=0.0,
                 tol=1e-4,
                 maxiter=300,
                 pickable=True,
                 noise=True,
                 weighted_noise=False,
                 run_analysis=True):
        modes = 'all', 'each'
        if mode not in modes:
            raise ValueError("The only modes implemented are {0}.".format(
                    strenum(modes, 'and')))
        self.pickable = pickable
        if isinstance(acquisition_or_rec_map, QubicAcquisition):
            acquisition = acquisition_or_rec_map
            convolution = acquisition.get_convolution_peak_operator()
            self._scene_band = acquisition.scene.nu / 1e9
            self._scene_nside = acquisition.scene.nside
            self._scene_kind = acquisition.scene.kind
            if not pickable:
                self.acquisition = acquisition
            self._detectors_number = len(acquisition.instrument.detector)
            if run_analysis:
                self._tod, self.input_map_convolved = map2tod(
                    acquisition, input_map, convolution=True)
                if noise:
                    self._tod += acquisition.get_noise()
                if mode == 'all':
                    tod2map_func = tod2map_all
                elif mode == 'each':
                    tod2map_func = tod2map_each
                self.reconstructed_map, self.coverage = tod2map_func(
                    acquisition, self._tod, tol=tol, maxiter=maxiter)
            self.calc_coverage()
            self.coverage = convolution(self.coverage)
            self.coverage_thr = coverage_thr
        else:
            if coverage == None:
                raise TypeError("Must provide the coverage map!")
            self._coverage_thr = coverage_thr
            self._scene_band = None
            self._scene_nside = hp.npix2nside(len(coverage))
            self._scene_kind = 'IQU' if acquisition_or_rec_map.shape[1] == 3 else 'I'
            self._detectors_number = 1984 if self._scene_kind == 'IQU' else 992
            self.input_map_convolved = input_map # here the input map must be already convolved!
            self.reconstructed_map = acquisition_or_rec_map
            self.coverage = coverage

    @property
    def coverage_thr(self):
        return self._coverage_thr

    @coverage_thr.setter
    def coverage_thr(self, thr):
        if thr < 0.0 or thr > 1.0:
            raise ValueError("Coverage threshold must be between 0. and 1.")
        self._coverage_thr = thr
        
    @property
    def input_map_convolved(self):
        return self._input_map_convolved

    @input_map_convolved.setter
    def input_map_convolved(self, map):
        self._input_map_convolved = map

    @property
    def reconstructed_map(self):
        return self._reconstructed_map

    @reconstructed_map.setter
    def reconstructed_map(self, map):
        self._reconstructed_map = map

    @property
    def coverage(self):
        return self._coverage

    @coverage.setter
    def coverage(self, map):
        self._coverage = map

    @property
    def diff_map(self):
        return self.reconstructed_map - self.input_map_convolved

    @property
    def chi2_map(self):
        return self.diff_map**2

    def chi2(self, bin_width = 0.01):
        if self._scene_kind == 'I':
            return self._chi2(self.chi2_map, bin_width)
        else:
            chi2 = np.empty(3. / bin_width)
            for i in xrange(3):
                chi2[i / bin_width : (i+1)/bin_width] = \
                  self._chi2(self.chi2_map[..., i], bin_width)
            mask = chi2 <= 0
            chi2[mask] = None
            return chi2.reshape(3, 1. / bin_width).T
                    
    def _chi2(self, map, bin_width):
        map_, cov_ = self.mask_map(map)
        chi2 = np.empty(1. / bin_width)
        for bin in np.arange(0., 1., bin_width):
            bin_mask = (cov_ > bin) * (cov_ < bin + bin_width)
            chi2[bin / bin_width] = map_[bin_mask].sum()
        return chi2

    def Omega(self):
        return (self.normalized_coverage().sum() *
                hp.pixelfunc.nside2pixarea(self._scene_nside, degrees=True))

    def Eta(self):
        return (self.Omega() / 
          ((hp.ud_grade(self.normalized_coverage(), 32) *
            hp.pixelfunc.nside2pixarea(32, degrees=True))**2).sum())

    def OneDetCoverage(self, detnum, convolve=False, nside=None):
        '''
        analysis.OneDetCoverage(detnum, convolve=False, nside=None) -> coverage map for one detector
        If convolve equals True, then return convolved map
        If nside != None, return map with the given nside.
        '''
        if nside == None:
            nside = self._scene_nside
        acq = copy(self.acquisition)
        acq.instrument = acq.instrument[detnum]
        projection = acq.get_projection_operator()
        coverage_4_curr_det = hp.ud_grade(projection.pT1(), nside)
        if convolve:
            convolution = acq.get_convolution_peak_operator()
            coverage_4_curr_det = convolution(coverage_4_curr_det)
            coverage_4_curr_det[coverage_4_curr_det < 0.] = 0.
        return coverage_4_curr_det

    def Lambda(self, ndet=None, nsamples=1, nside=None, seed=None):
        '''
        analysis.Lambda(ndet=ND, nsamples=NS) -> overlapping of QUBIC detectors
        ndet - number of detectors in sample.
               If None, then all detectors are taken and 
               nsamples=1
        nsamples - number of ndet samples for calculating average overlapping
        result is in (0, 1] range, where 1 meens perfect overlapping0
        '''
        if nside == None:
            nside = self._scene_nside
        if not self.pickable:
            cov_sum = np.zeros(hp.nside2npix(nside))
            cov_prod = np.ones(hp.nside2npix(nside))
            if not ndet:
                nsamples = 1
            if not isinstance(nsamples, int) or nsamples < 1:
                raise ValueError("Number of samples must be positive integer")
            overlapping = 0
            for isample in xrange(nsamples):
                if ndet:
                    random.seed(seed) # when seed==None this method takes the system time for seed                        
                    index = np.arange(self._detectors_number)
                    index = random.sample(index, ndet)
                else:
                    ndet = self._detectors_number
                    index = np.arange(ndet)
                for i in index:
                    coverage_4_curr_det = self.OneDetCoverage(i, convolve=True, nside=nside)
                    coverage_4_curr_det /= coverage_4_curr_det.max()
                    cov_sum += coverage_4_curr_det
                    cov_prod *= coverage_4_curr_det
                cov_prod = np.power(cov_prod, 1/float(ndet))
                cov_sum /= ndet
                overlapping += cov_prod.sum() / cov_sum.sum()
            return overlapping / nsamples

    def normalized_coverage(self):
        cov = self.coverage.copy()
        cov /= cov.max()
        mask = cov < self.coverage_thr
        cov[mask] = 0.
        return cov

    def mask_map(self, map):
        cov = self.normalized_coverage()
        map[cov == 0.] = 0.
        return map, cov

    def calc_coverage(self, nside=None):
        if hasattr(self, 'coverage'):
            return
        if not hasattr(self, 'acquisition'):
            raise ValueError('myqubic.QubicAnalysis.calc_coverage:' +
                             ' For calculating coverage the analysis ' +
                             'object has to have acquisition in attributes!')
        ndet = len(self.acquisition.instrument)
        if nside == None:
            nside = self._scene_nside
        coverage = np.zeros(hp.nside2npix(nside))
        for idet in xrange(ndet):
            coverage += self.OneDetCoverage(idet, convolve=False)
        self.coverage = cov
