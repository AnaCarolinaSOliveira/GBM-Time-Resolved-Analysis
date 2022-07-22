#!/usr/bin/env python3 

import gbm
from gbm.data import TTE, GbmDetectorCollection
from gbm.binning.unbinned import bin_by_time
from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
import matplotlib.pyplot as plt

from gbm.data import RSP
from gbm.spectra.fitting import SpectralFitterPgstat, SpectralFitterPstat
from gbm.spectra.functions import SmoothlyBrokenPowerLaw, PowerLaw, Comptonized, Band
from gbm.plot import ModelFit

import os 
import numpy as np
import numexpr
import json
from math import isclose

#Parameters 
# TODO function to get some of this information automatically, especially the background range 
EVENT = '080916009'
DETECTORS = '00011000000010'
BINNING = 1.024
BKG_RANGE =  [(-18.0, 0.0), (100.0, 260.0)] 
ERANGE_NAI = (8.0, 900.0)
ERANGE_BGO = (240, 38000.0)
ORDER = 2
STATS = ['pstat', 'pgstat']
MODELS = {'band': Band(), 'pl': PowerLaw(), 'comp': Comptonized(), 'sbpl': SmoothlyBrokenPowerLaw()}
PATH = '/Users/anacarolinaoliveira/Documents/Documents_MacBook/INFN/Data/GRB{}_GBM/current'.format(EVENT)
PATH_RESULTS = '/Users/anacarolinaoliveira/Documents/Documents_MacBook/INFN/time_resolved'
TIME_RANGE = (-10, 80) #time range of event that we want to do the bayesian block binning
p0 = 0.01

# TODO implement the filename variable (v01 etc)
def detector_list():
    keys = {0: 'n0', 1: 'n1', 2: 'n2', 3: 'n3', 4: 'n4', 5: 'n5', 6: 'n6',
            7: 'n7', 8: 'n8', 9: 'n9', 10: 'na', 11: 'nb', 12: 'b0', 13: 'b1'}
    list_dtc = list()
    for i in range(len(DETECTORS)):
        if DETECTORS[i] == '1':
            list_dtc.append(keys[i])
    return list_dtc

def get_fit_file(detector):
    for file in os.listdir(PATH):
        if file.startswith('glg_tte_{}_bn{}'.format(detector, EVENT)) and file.endswith('.fit'):
            filename = file
    return filename

def get_rsp_file(detector):
    for file in os.listdir(PATH):
        if file.startswith('glg_cspec_{}_bn{}'.format(detector, EVENT)) and file.endswith('.rsp2'):
            filename = file
            break
        elif file.startswith('glg_cspec_{}_bn{}'.format(detector, EVENT)) and file.endswith('.rsp'):
            filename = file
    return filename

def get_brightest_det():
    list_dtc = detector_list()
    counts = {}
    for detector in list_dtc:
        if detector.startswith('n'):
            filename = get_fit_file(detector)
            globals()['{}'.format(detector)] = TTE.open('{}/{}'.format(PATH, filename))
            counts['{}'.format(detector)] = globals()['{}'.format(detector)].data.size
    brightest = max(counts, key=counts.get)
    return brightest

def tte2phaiis():
    list_dtc = detector_list()
    phaii_list = list()
    for detector in list_dtc:
        filename = get_fit_file(detector)
        globals()['{}'.format(detector)] = TTE.open('{}/{}'.format(PATH, filename))
        globals()['{}_phaii'.format(detector)] = globals()['{}'.format(detector)].to_phaii(bin_by_time, BINNING, time_ref=0.0)
        phaii_list.append(globals()['{}_phaii'.format(detector)])
    phaiis = GbmDetectorCollection.from_list(phaii_list)
    return phaiis

def gen_bkg(phaiis, order_pol):
    backfitters = [BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=BKG_RANGE) for phaii in phaiis]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=phaiis.detector())
    backfitters.fit(order=order_pol)
    bkgds = backfitters.interpolate_bins(phaiis.data()[0].tstart, phaiis.data()[0].tstop)
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=phaiis.detector())
    return bkgds 

def rsp_list():
    rsp_list = list()
    list_dtc = detector_list()
    for i in range(len(list_dtc)):
        rsp_file = get_rsp_file(list_dtc[i])
        globals()['rsp{}'.format(i)] = RSP.open('{}/{}'.format(PATH, rsp_file))
        rsp_list.append(globals()['rsp{}'.format(i)])
        rsps = GbmDetectorCollection.from_list(rsp_list)
    return rsps

def plotting(dataset, bins):
    plt.figure(figsize=(15, 8))
    plt.hist(dataset, bins=150, histtype='stepfilled', alpha=0.5, density=True, label='Standard Histogram')
    plt.hist(dataset, bins=bins, histtype='step', density=True, label='Bayesian blocks', color='black')
    plt.legend(loc='upper right')
    plt.title('GRB{}'.format(EVENT))
    plt.savefig('{}/GRB{}/plots/lightcurve_GRB{}.pdf'.format(PATH_RESULTS, EVENT, EVENT))
    plt.show()
 
def bayesian_blocks(tt, ttstart, ttstop, p0, bkg_integral_distribution=None):
    """
    Source: bin_by_bayesian_blocks() function from threeML framework. 
    (Copyright 2017--2021, G.Vianello, J. M. Burgess, N. Di Lalla, N. Omodei, H. Fleischhack. Revision fe390c3e.)

    Divide a series of events characterized by their arrival time in blocks
    of perceptibly constant count rate. If the background integral distribution
    is given, divide the series in blocks where the difference with respect to
    the background is perceptibly constant.

    :param tt: arrival times of the events
    :param ttstart: the start of the interval
    :param ttstop: the stop of the interval
    :param p0: the false positive probability. This is used to decide the penalization on the likelihood, so this
    parameter affects the number of blocks
    :param bkg_integral_distribution: (default: None) If given, the algorithm account for the presence of the background and
    finds changes in rate with respect to the background
    :return: the np.array containing the edges of the blocks
    """

    # Verify that the input array is one-dimensional
    tt = np.asarray(tt, dtype=float)
    assert tt.ndim == 1

    if bkg_integral_distribution is not None:
        # Transforming the inhomogeneous Poisson process into an homogeneous one with rate 1,
        # by changing the time axis according to the background rate
        logger.debug(
            "Transforming the inhomogeneous Poisson process to a homogeneous one with rate 1..."
        )
        t = np.array(bkg_integral_distribution(tt))
        logger.debug("done")

        # Now compute the start and stop time in the new system
        tstart = bkg_integral_distribution(ttstart)
        tstop = bkg_integral_distribution(ttstop)
    else:
        t = tt
        tstart = ttstart
        tstop = ttstop

    # Create initial cell edges (Voronoi tessellation)
    edges = np.concatenate([[t[0]], 0.5 * (t[1:] + t[:-1]), [t[-1]]])

    # Create the edges also in the original time system
    edges_ = np.concatenate([[tt[0]], 0.5 * (tt[1:] + tt[:-1]), [tt[-1]]])

    # Create a lookup table to be able to transform back from the transformed system
    # to the original one
    lookup_table = {key: value for (key, value) in zip(edges, edges_)}

    # The last block length is 0 by definition
    block_length = tstop - edges

    if np.sum((block_length <= 0)) > 1:

        raise RuntimeError(
            "Events appears to be out of order! Check for order, or duplicated events."
        )

    N = t.shape[0]

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # eq. 21 from Scargle 2012
    prior = 4 - np.log(73.53 * p0 * (N ** -0.478))

    # This is where the computation happens. Following Scargle et al. 2012.
    # This loop has been optimized for speed:
    # * the expression for the fitness function has been rewritten to
    #  avoid multiple log computations, and to avoid power computations
    # * the use of scipy.weave and numexpr has been evaluated. The latter
    #  gives a big gain (~40%) if used for the fitness function. No other
    #  gain is obtained by using it anywhere else

    # Set numexpr precision to low (more than enough for us), which is
    # faster than high
    oldaccuracy = numexpr.set_vml_accuracy_mode("low")
    numexpr.set_num_threads(1)
    numexpr.set_vml_num_threads(1)

    # Speed tricks: resolve once for all the functions which will be used
    # in the loop
    numexpr_evaluate = numexpr.evaluate
    numexpr_re_evaluate = numexpr.re_evaluate

    # Pre-compute this

    aranges = np.arange(N + 1, 0, -1)
    
    for R in range(N):
        br = block_length[R + 1]
        T_k = (
            block_length[: R + 1] - br
        )  # this looks like it is not used, but it actually is,
        # inside the numexpr expression

        # N_k: number of elements in each block
        # This expression has been simplified for the case of
        # unbinned events (i.e., one element in each block)
        # It was:
        # N_k = cumsum(x[:R + 1][::-1])[::-1]
        # Now it is:
        N_k = aranges[N - R :]
        # where aranges has been pre-computed

        # Evaluate fitness function
        # This is the slowest part, which I'm speeding up by using
        # numexpr. It provides a ~40% gain in execution speed.

        # The first time we need to "compile" the expression in numexpr,
        # all the other times we can reuse it

        if R == 0:
            fit_vec = numexpr_evaluate(
                """N_k * log(N_k/ T_k) """,
                optimization="aggressive",
                local_dict={"N_k": N_k, "T_k": T_k},
            )
        else:
            fit_vec = numexpr_re_evaluate(local_dict={"N_k": N_k, "T_k": T_k})

        A_R = fit_vec - prior  # type: np.ndarray
        A_R[1:] += best[:R]
        i_max = A_R.argmax()
        last[R] = i_max
        best[R] = A_R[i_max]

    numexpr.set_vml_accuracy_mode(oldaccuracy)

    # Now peel off and find the blocks (see the algorithm in Scargle et al.)
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N

    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]

    change_points = change_points[i_cp:]
    edg = edges[change_points]

    # Transform the found edges back into the original time system
    if bkg_integral_distribution is not None:
        final_edges = [lookup_table[x] for x in edg]
    else:
        final_edges = edg

    # Now fix the first and last edge so that they are tstart and tstop
    final_edges[0] = ttstart
    final_edges[-1] = ttstop

    return np.asarray(final_edges)

def main():
    #preparing data for bblocks binning step
    brightest = get_brightest_det()
    filename = get_fit_file(brightest)
    bright_det = TTE.open('{}/{}'.format(PATH, filename))
    sliced = bright_det.slice_time([TIME_RANGE])
    t = np.array(sliced.data.time)
    
    #getting bayesian blocks 
    edges = bayesian_blocks(t, TIME_RANGE[0], TIME_RANGE[1], p0)
    bins = edges.tolist()
    
    #creating dir to save results and plots:
    if not os.path.exists('{}/GRB{}'.format(PATH_RESULTS, EVENT)):
        os.mkdir('{}/GRB{}'.format(PATH_RESULTS, EVENT))
    if not os.path.exists('{}/GRB{}/plots'.format(PATH_RESULTS, EVENT)):
        os.mkdir('{}/GRB{}/plots'.format(PATH_RESULTS, EVENT))
        
    plotting(t, bins)

    #generating phaii and bkg files for all detectors 
    phaiis = tte2phaiis()
    bkgds = gen_bkg(phaiis, ORDER)
    
    #dictionary to store results
    results = {}
    
    #now, start iterating through all intervals
    for interval in range(len(bins)-1):
        results['int_{}'.format(interval)] = {}
        try:
            source_range = (bins[interval], bins[interval + 1]) #source range is going to become the slice we're analyzing
        except IndexError:
            source_range = (bins[interval], bins[TIME_RANGE[1]])
        
        phas = phaiis.to_pha(time_ranges=source_range, nai_kwargs={'energy_range':ERANGE_NAI}, bgo_kwargs={'energy_range':ERANGE_BGO})
        rsps = rsp_list() 

        # Interpolate response files to get DRMs at center of the source window
        #rsps_interp = [rsp.interpolate(pha.tcent) for rsp, pha in zip(rsps, phas)]

        # Initialize model with our PHAs, backgrounds, and responses
        for stat in STATS:
            results['int_{}'.format(interval)]['{}'.format(stat)] = {}
            for model in MODELS:
                results['int_{}'.format(interval)]['{}'.format(stat)]['{}'.format(model)] = {}
                if stat == 'pstat':
                    specfitter = SpectralFitterPstat(phas, bkgds.to_list(), rsps.to_list(), method='TNC')
                else:
                    specfitter = SpectralFitterPgstat(phas, bkgds.to_list(), rsps.to_list(), method='TNC')
                    
                specfitter.fit(MODELS[model], options={'maxiter': 1000})
                
                if specfitter.statistic != 0.0:
                    modelplot = ModelFit(fitter=specfitter)
                    plt.savefig('{}/GRB{}/plots/modelfit_int{}_{}_{}.pdf'.format(PATH_RESULTS, EVENT, interval, stat, model))
                
                #adding results to dictionary 
                results[f'int_{interval}'][f'{stat}'][f'{model}']['message'] = specfitter.message
                results[f'int_{interval}'][f'{stat}'][f'{model}']['stat/dof'] = '{}/{}'.format(specfitter.statistic, specfitter.dof)
                results[f'int_{interval}'][f'{stat}'][f'{model}']['amplitude'] = specfitter.parameters[0]
                try:
                    results[f'int_{interval}'][f'{stat}'][f'{model}']['amplitude_err-'] = specfitter.asymmetric_errors(cl=0.9)[0][0]
                    results[f'int_{interval}'][f'{stat}'][f'{model}']['amplitude_err+'] = specfitter.asymmetric_errors(cl=0.9)[0][1]
                except RuntimeError:
                    continue
                if model == 'band' or model == 'sbpl':
                    results[f'int_{interval}'][f'{stat}'][f'{model}']['epeak'] = specfitter.parameters[1]
                    results[f'int_{interval}'][f'{stat}'][f'{model}']['alpha'] = specfitter.parameters[2]
                    results[f'int_{interval}'][f'{stat}'][f'{model}']['beta'] = specfitter.parameters[3]
                    try:
                        results[f'int_{interval}'][f'{stat}'][f'{model}']['epeak_err-'] = specfitter.asymmetric_errors(cl=0.9)[1][0]
                        results[f'int_{interval}'][f'{stat}'][f'{model}']['epeak_err+'] = specfitter.asymmetric_errors(cl=0.9)[1][1]
                        results[f'int_{interval}'][f'{stat}'][f'{model}']['alpha_err-'] = specfitter.asymmetric_errors(cl=0.9)[2][0]
                        results[f'int_{interval}'][f'{stat}'][f'{model}']['alpha_err+'] = specfitter.asymmetric_errors(cl=0.9)[2][1]
                        results[f'int_{interval}'][f'{stat}'][f'{model}']['beta_err-'] = specfitter.asymmetric_errors(cl=0.9)[3][0]
                        results[f'int_{interval}'][f'{stat}'][f'{model}']['beta_err+'] = specfitter.asymmetric_errors(cl=0.9)[3][1]
                    except RuntimeError:
                        continue
                elif model == 'pl':
                    results[f'int_{interval}'][f'{stat}']['pl']['alpha'] = specfitter.parameters[1]
                    try:
                        results[f'int_{interval}'][f'{stat}']['pl']['alpha_err-'] = specfitter.asymmetric_errors(cl=0.9)[1][0]
                        results[f'int_{interval}'][f'{stat}']['pl']['alpha_err+'] = specfitter.asymmetric_errors(cl=0.9)[1][1]
                    except RuntimeError:
                        continue
                elif model == 'comp':
                    results[f'int_{interval}'][f'{stat}']['comp']['epeak'] = specfitter.parameters[1]
                    results[f'int_{interval}'][f'{stat}']['comp']['alpha'] = specfitter.parameters[2]
                    try:
                        results[f'int_{interval}'][f'{stat}']['comp']['epeak_err-'] = specfitter.asymmetric_errors(cl=0.9)[1][0]
                        results[f'int_{interval}'][f'{stat}']['comp']['epeak_err+'] = specfitter.asymmetric_errors(cl=0.9)[1][1]
                        results[f'int_{interval}'][f'{stat}']['comp']['alpha_err-'] = specfitter.asymmetric_errors(cl=0.9)[2][0]
                        results[f'int_{interval}'][f'{stat}']['comp']['alpha_err+'] = specfitter.asymmetric_errors(cl=0.9)[2][1]
                    except RuntimeError:
                        continue
    
    results_file = json.dumps(results, indent = 4)
    with open('{}/{}/modelfit_results.json'.format(PATH_RESULTS, EVENT), 'w') as outfile:
        outfile.write(results_file)

if __name__ == "__main__":
    main()
