import numpy as np
from tqdm import tqdm

def write_freqs_fit_file(output_tag, freqs, freq_errs, times):
    """
    write frequencies to a file
    """
    with open(output_tag + '.freqs', 'w') as myf:
        for freq, fe, t in zip(freqs, freq_errs, times):
            print(f"{t} {freq} {fe}", file=myf)
    return np.asarray(freqs), np.asarray(freq_errs)

def write_tim_file(output_tag, toas, toa_errors, pn=None, observatory="BAT"):
    """
    print out a fake .tim file
    """
    with open(output_tag + '.tim', 'w') as myf:
        print("FORMAT 1", file=myf)
        print("MODE 1", file=myf)
        if pn is not None:
            for toa, terr, pn in zip(toas, toa_errors, pn):
                print(f"fake 1000 {toa} {terr*1e6} {observatory} -pn {pn}", file=myf)
        else:
            for toa, terr in zip(toas, toa_errors):
                print(f"fake 1000 {toa} {terr*1e6} {observatory}", file=myf)

def write_par(parfile, F0, F1, PEPOCH, F2=None, fit_F2=False, psrj="FAKE", F1err=1e-13, F0err=1e-7, fit_omgdot=True, raj="00:00", decj="00:00", fit_pos=False):
    """
    write out a fake par file
    """
    with open(parfile + ".par", 'w') as myf:
        print(f"{'PSRJ':15}{psrj}", file=myf)
        print(f"{'RAJ':15} {raj} {1 if fit_pos else 0}", file=myf)
        print(f"{'DECJ':15} {decj} {1 if fit_pos else 0}", file=myf)
        print(f"{'F0':15}{F0} 1  {F0err}", file=myf)
        if fit_omgdot:
            fit=1
        else:
            fit=0
        if F1 is not None:
            print(f"{'F1':15}{F1:.10e} {fit} {F1err}", file=myf)
        if F2 is not None:
            print(f"{'F2':15}{F2:.10e} {int(fit_F2)}", file=myf)
        print(f"{'PEPOCH':15}{PEPOCH}", file=myf)
        print("TRACK -2", file=myf)

def fit_toas_to_get_frequencies(toas, toa_errors, F0, F1, PEPOCH,
        Ntoas_per_fit=3, tmpdir='./'):
    """
    Fit TOAs to get frequencies.
    """
    # do import here so that it doesn't necessarily
    # break things for people
    # unless they try to run this function.
    import libstempo
    freqs_fit = []
    freqs_errs_fit = []
    F1s_fit = []
    F1s_errs_fit = []
    times_fit = []
    for ii in tqdm(range(Ntoas_per_fit, toas.size)):
        idxs = np.arange(ii-Ntoas_per_fit, ii)
        # write small tim file
        write_tim_file(tmpdir + '/tmp', toas[idxs], toa_errors[idxs])
        # load up and do fit
        PEPOCH_tmp = toas[idxs[int((Ntoas_per_fit-1)/2)]]
        # if ii == Ntoas_per_fit:
        write_par(tmpdir + '/tmp', F0 + F1*(PEPOCH_tmp - PEPOCH)*86400, F1, PEPOCH_tmp)
        # else:
        #     write_par('tmp', psr['F0'].val + psr['F1'].val*(PEPOCH_tmp -
        #         psr['PEPOCH'].val)*86400, psr['F1'].val, PEPOCH_tmp,
        #         F1err=psr['F1'].err, F0err=psr['F0'].err)
        psr = libstempo.tempopulsar(parfile=tmpdir+'/tmp.par', timfile=tmpdir+'/tmp.tim')
        psr.fit()
        freqs_fit.append(psr['F0'].val)
        freqs_errs_fit.append(psr['F0'].err)
        F1s_fit.append(psr['F1'].val)
        F1s_errs_fit.append(psr['F1'].err)
        times_fit.append(PEPOCH_tmp)
    return np.array(freqs_fit).astype(float), np.array(freqs_errs_fit).astype(float), np.array(times_fit).astype(float)
def fit_toas_to_get_F0_F1(toas, toa_errors, F0, F1, PEPOCH, Ntoas_per_fit=3,
        tmpdir='./'):
    """
    Fit TOAs to get frequencies.
    """
    # do import here so that it doesn't necessarily
    # break things for people
    # unless they try to run this function.
    import libstempo
    freqs_fit = []
    freqs_errs_fit = []
    F1s_fit = []
    F1s_errs_fit = []
    times_fit = []
    cov = []
    for ii in tqdm(range(Ntoas_per_fit, toas.size)):
        idxs = np.arange(ii-Ntoas_per_fit, ii)
        # write small tim file
        write_tim_file(tmpdir + '/tmp', toas[idxs], toa_errors[idxs])
        # load up and do fit
        PEPOCH_tmp = toas[idxs[int((Ntoas_per_fit-1)/2)]]
        # if ii == Ntoas_per_fit:
        write_par(tmpdir + '/tmp', F0 + F1*(PEPOCH_tmp - PEPOCH)*86400, F1, PEPOCH_tmp)
        # else:
        #     write_par('tmp', psr['F0'].val + psr['F1'].val*(PEPOCH_tmp -
        #         psr['PEPOCH'].val)*86400, psr['F1'].val, PEPOCH_tmp,
        #         F1err=psr['F1'].err, F0err=psr['F0'].err)
        psr = libstempo.tempopulsar(parfile=tmpdir + '/tmp.par', timfile=tmpdir + '/tmp.tim')
        out = psr.fit()
        freqs_fit.append(psr['F0'].val)
        freqs_errs_fit.append(psr['F0'].err)
        F1s_fit.append(psr['F1'].val)
        F1s_errs_fit.append(psr['F1'].err)
        times_fit.append(PEPOCH_tmp)
        cov.append(out[2][1:3, 1:3])
    output_vals = np.c_[np.array(freqs_fit), np.array(F1s_fit)]
    output_errs = np.c_[np.array(freqs_errs_fit), np.array(F1s_errs_fit)]
    cov = np.array(cov)

    return np.array(times_fit).astype(float), output_vals.astype(float), output_errs.astype(float), cov.astype(float)

