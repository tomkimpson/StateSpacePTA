import sdeint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import argparse
import random
from scipy.optimize import minimize
from baboo.utils import fit_toas_to_get_frequencies, write_par, write_tim_file, write_freqs_fit_file
from baboo.simulation import OneComponentModelSim

def run(params):
    libstempo_in_use = False
    try:
        import libstempo
        libstempo_in_use = True
    except:
        print("No libstempo...")
    # libstempo_in_use = False

    print('simulate from known PSR...resetting start frequency, fdot, and PEPOCH')
    newpsr = libstempo.tempopulsar(parfile=params.par_file, timfile=params.tim_file)
    # get pets and sort them
    pets = np.sort(newpsr.pets())
    # change to seconds, round off to nearest second
    tstarts = np.round((pets - pets[0]) * 86400)
    # start 1 day later
    tstarts += 86400
    toa_errors = newpsr.toaerrs * 1e-6
    PEPOCH = newpsr['PEPOCH'].val
    pets0 = pets[0]
    # set omega dot at start time
    Textrap = (pets0 - newpsr['PEPOCH'].val) * 86400

    # omega(t_0) = F(pepoch) + Fdot(pepoch) * (t_0 - PEPOCH) + Fddot(pepoch)*(t_0 - PEPOCH)^2 / 2.
    omgc0 = newpsr['F0'].val + newpsr['F1'].val * Textrap + params.omgc_ddot * 0.5 * Textrap**2
    omgc_dot = newpsr['F1'].val + params.omgc_ddot * Textrap
    omgc_pepoch = newpsr['F0'].val

    write_par(params.output_tag + '.par', newpsr['F0'].val,
            newpsr['F1'].val, PEPOCH,
              F0err=1e-3, F1err=np.abs(omgc_dot/3))

    # values at start time

    phi0 = random.random()
    p0 = np.asarray([phi0, omgc0, omgc_dot])

    model_class = OneComponentModelSim(F2=params.omgc_ddot, Q_f=params.xi_phase**2, Q_f1=params.xi_freq**2)
    toas, toa_errors, states = model_class(pets, p0, toa_errors)

    # fit toas to get frequencies
    freqs_fit, freqs_errs_fit, times_fit = fit_toas_to_get_frequencies(toas,
            toa_errors, omgc0, omgc_dot, toas[0], Ntoas_per_fit=params.Ntoas_per_fit)

    # save file with fitted frequencies
    freqs, freq_errors = write_freqs_fit_file(params.output_tag,
                                              freqs_fit, freqs_errs_fit,
                                              times_fit
                                              )

    # save file with "true" frequencies
    write_freqs_fit_file(params.output_tag + '_true', states[0, :],
                         np.ones(freqs.size)*1e-10,
                         toas
                         )

    # TODO include file with true frequency derivatives as well.

    # plot
    plt.errorbar(times_fit, freqs, yerr=freq_errors, fmt='x')
    plt.errorbar(toas, states[0, :], fmt='o')
    plt.xlabel('MJD')
    plt.ylabel('$f(t) [Hz]$')
    plt.savefig(f"{params.output_tag}_frequencies_from_fits")
    plt.close()

    # save tim file
    write_tim_file(params.output_tag, toas, toa_errors)

    # read in from libstempo, now with new tim file
    psr = libstempo.tempopulsar(parfile=params.par_file, timfile=params.output_tag + '.tim')
    # fit
    psr.fit(iters=30)
    residuals = psr.residuals()

    # save output par
    psr.savepar(params.output_tag + '.par')

    # get libstempo fit residuals
    plt.errorbar(toas, residuals * omgc0, yerr=toa_errors * omgc0, fmt='x', label='libstempo')
    plt.legend()
    plt.xlim(toas[0], toas[-1])
    plt.savefig(params.output_tag + 'residuals')
    plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi_freq", default=1e-11, type=float, help='white noise in frequrency derivative')
    parser.add_argument("--xi_phase", default=0, type=float, help='white noise in phase derivatives')
    parser.add_argument("--xi_fdot", default=0, type=float, help='white noise in second derivative of frequency')
    parser.add_argument("--xi_fddot", default=0, type=float, help='white noise in third derivative of frequency')

    parser.add_argument("--omgc-ddot", default=0, type=float, help="starting second derivative")
    parser.add_argument("--output-tag", default='out', type=str, help='output file tag')
    parser.add_argument("--tim-file", default=None, type=str, help="simulate based on observing cadence", required=True)
    parser.add_argument("--par-file", default=None, type=str, help="simulate based on observing cadence", required=True)
    parser.add_argument("--Ntoas-per-fit", default=3, type=int, help="number of per fit")
    params = parser.parse_args()
    run(params)
