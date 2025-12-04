import erfa
import warnings
warnings.filterwarnings('ignore', category=erfa.ErfaWarning, append=True)

import numpy as np
import pylab as plt
import time
import pickle
from bagle import sensitivity

from bagle import model
from astropy.table import Table
from astropy.time import Time
from flystar import motion_model


def make_roman_lightcurve(mod, t, mod_filt_idx=0, filter_name='F146',
                          noise=True, tint=57, verbose=False, plot=True, 
                          zoom_tE_val = 3., time_window = 3.,
                          outdir='lightcurves/', outfile_suffix=''):
    """
    Given a BAGLE model, generate photometric and astrometric data

    Parameters
    ----------
    mod : model.ModelABC
        BAGLE model instance.
    t : numpy.array
        Array of times in MJD to sample the photometry and astrometry at.

    Optional Parameters
    -------------------
    mod_filt_idx : int
        The index of the filter on the model to query.
    filter_name : str
        The name of the Roman filter. Used to set zeropoints.
    tint : float
        The integration time of the individual time stamps.
    verbose : bool
        Print out quantities along the way.
    plot : bool
        Make plots of the photometry and astrometry lightcurves.
    zoom_tE_val : float
        When plotting zoomed in values, indicates range time tE to plot.
        (i.e. zoom_tE_val = 3 plots (-3tE + t0, 3tE + t0)).
    time_window : float
        When plotting time average astrometry controls window in DAYS to
        average over.

    Output
    ------
    data_table : astropy.table
        Astropy table containing magnitudes and astrometric positions over time.
        Table columns are t, x, y, xe, ye with all positions in milli-arcseconds
        and time in MJD.
    
    """
    # Get all synthetic magnitude and positions from the model object.
    try:
        img, amp = mod.get_all_arrays(t, filt_idx=mod_filt_idx)
        mag = mod.get_photometry(t, filt_idx=mod_filt_idx, amp_arr=amp)
        ast = 1e3 * mod.get_astrometry(t, filt_idx=mod_filt_idx, image_arr=img, amp_arr=amp)

        mag = mag.reshape(len(t))
    except:
        mag = mod.get_photometry(t, filt_idx=mod_filt_idx)
        ast = 1e3 * mod.get_astrometry(t, filt_idx=mod_filt_idx)

    mag_err, ast_err = get_roman_noise(mag, tint, filter_name)

    if noise:
        mag += np.random.normal(0, mag_err)

        ast_err = np.array([ast_err, ast_err]).T
        ast += np.random.normal(size=ast.shape) * ast_err

    if verbose:
        print(f'Mean {filter_name} Mag = {mag.mean():.1f} +/- {mag_err.mean():.2f} mag')
        print(f'Mean {filter_name} ast err = {ast_err.mean():.2f} mas')


    ##
    ## Make an output table.
    ##
    tab = Table((t, mag, mag_err),
                     names=(f't_{filter_name}', f'm_{filter_name}', f'me_{filter_name}'))

    tab[f'x_{filter_name}'] = ast[:, 0]
    tab[f'y_{filter_name}'] = ast[:, 1]
    tab[f'xe_{filter_name}'] = ast_err[:, 0]
    tab[f'ye_{filter_name}'] = ast_err[:, 1]

    ##
    ## Plot
    ##
    if plot:
        # Determine the lensed - unlensed astrometry residuals.
        ast_unlensed = mod.get_astrometry_unlensed(t, filt_idx=mod_filt_idx) * 1e3
        ast_resid = ast - ast_unlensed

        
        # Also make a model curve with 0.5 day time sampling.
        t_mod = np.arange(t.min(), t.max(), 0.5)
        try:
            img_mod, amp_mod = mod.get_all_arrays(t_mod, filt_idx=mod_filt_idx)
            mag_mod = mod.get_photometry(t_mod, filt_idx=mod_filt_idx, amp_arr=amp)
            ast_mod = 1e3 * mod.get_astrometry(t_mod, filt_idx=mod_filt_idx, image_arr=img, amp_arr=amp)

            mag_mod = mag_mod.reshape(len(t_mod))
        except:
            mag_mod = mod.get_photometry(t_mod, filt_idx=mod_filt_idx)
            ast_mod = 1e3 * mod.get_astrometry(t_mod, filt_idx=mod_filt_idx)
        
        ast_unlensed_mod = mod.get_astrometry_unlensed(t_mod, filt_idx=mod_filt_idx) * 1e3
        ast_resid_mod = ast_mod - ast_unlensed_mod
        

        #time_window = 3 # days
        t_bin, ast_x_res_bin, ast_xe_res_bin = moving_average(t, ast_resid[:, 0], time_window, errors=tab[f'xe_{filter_name}'])
        t_bin, ast_y_res_bin, ast_ye_res_bin = moving_average(t, ast_resid[:, 1], time_window, errors=tab[f'ye_{filter_name}'])

        zoom_dt = [mod.t0 - zoom_tE_val*mod.tE, mod.t0 + zoom_tE_val*mod.tE]

        plt_msc = [['F1', 'AA', 'A1'],
                   ['F1', 'AA', 'A2'],
                   ['F2', 'BB', 'B1'],
                   ['F2', 'BB', 'B2']]
        fig, axs = plt.subplot_mosaic(plt_msc,
                                      figsize=(16, 8))
        plt.subplots_adjust(left=0.1, hspace=1.2, wspace=0.5, top=0.92)

        # Photometry vs. time
        axs['F1'].errorbar(t, mag, yerr=mag_err, label=filter_name,
                           ls='none', marker='.', alpha=0.2)
        axs['F1'].plot(t_mod, mag_mod, ls='-', marker='None', zorder=50)
        axs['F1'].set_ylabel(f'{filter_name} mag')
        axs['F1'].axvline(mod.t0, ls='--', color='grey')
        axs['F1'].invert_yaxis()

        # Photometry vs. time -- zoomed
        axs['F2'].errorbar(t, mag, yerr=mag_err, ls='none', marker='.', alpha=0.2)
        axs['F2'].plot(t_mod, mag_mod, ls='-', marker='None', zorder=50)
        axs['F2'].axvline(mod.t0, ls='-', color='grey')
        axs['F2'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
        axs['F2'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
        axs['F2'].set_ylabel(f'Zoomed\n {filter_name} mag')
        axs['F2'].set_xlabel('Time (MJD)')
        axs['F2'].invert_yaxis()
        axs['F2'].set_xlim(zoom_dt)

        # Astrometry on sky
        axs['AA'].errorbar(tab[f'x_{filter_name}'], tab[f'y_{filter_name}'],
                           xerr=tab[f'xe_{filter_name}'], yerr=tab[f'ye_{filter_name}'],
                           ls='none', marker='.', alpha=0.2)
        axs['AA'].plot(ast_mod[:, 0], ast_mod[:, 1], ls='-', marker='None', zorder=50)
        axs['AA'].set_xlabel(f'$\Delta\\alpha \cos \delta$ (mas)')
        axs['AA'].set_ylabel(f'$\Delta\delta$ (mas)')

        # Astrometry vs. time - East
        axs['A1'].errorbar(tab[f't_{filter_name}'], tab[f'x_{filter_name}'], yerr=tab[f'xe_{filter_name}'],
                           ls='none', marker='.', alpha=0.2)
        axs['A1'].plot(t_mod, ast_mod[:, 0], ls='-', marker='None', zorder=50)
        axs['A1'].set_xlabel(f'Time (MJD)')
        axs['A1'].set_ylabel(f'$\Delta\\alpha \cos \delta$\n(mas)')
        axs['A1'].sharex(axs['F1'])

        # Astrometry vs. time - North
        axs['A2'].errorbar(tab[f't_{filter_name}'], tab[f'y_{filter_name}'], yerr=tab[f'ye_{filter_name}'],
                           ls='none', marker='.', alpha=0.2)
        axs['A2'].plot(t_mod, ast_mod[:, 1], ls='-', marker='None', zorder=50)
        axs['A2'].set_xlabel(f'Time (MJD)')
        axs['A2'].set_ylabel(f'$\Delta\delta$\n(mas)')
        axs['A2'].sharex(axs['F1'])


        # Astrometry residuals on sky
        ast_res_rng = np.max([np.std(ast_x_res_bin), np.std(ast_y_res_bin), np.median(ast_xe_res_bin)]) * 3.0
        if ~np.isfinite(ast_res_rng):
            ast_res_rng = 1.0 # mas

        axs['BB'].errorbar(ast_x_res_bin, ast_y_res_bin,
                           xerr=ast_xe_res_bin, yerr=ast_ye_res_bin,
                           ls='none', marker='.', alpha=0.2)
        axs['BB'].plot(ast_resid_mod[:, 0], ast_resid_mod[:, 1], ls='-', marker='None', zorder=50)
        axs['BB'].set_xlabel(f'Lensing Signal\n $\Delta\\alpha \cos \delta$ (mas)')
        axs['BB'].set_ylabel(f'Lensing Signal\n $\Delta\delta$ (mas)')
        axs['BB'].set_xlim([-ast_res_rng, ast_res_rng])
        axs['BB'].set_ylim([-ast_res_rng, ast_res_rng])

        # Astrometry vs. time - East
        axs['B1'].errorbar(t_bin, ast_x_res_bin, yerr=ast_xe_res_bin,
                           ls='none', marker='.')
        axs['B1'].plot(t_mod, ast_resid_mod[:, 0], ls='-', marker='None', zorder=50)
        axs['B1'].set_ylim([-ast_res_rng, ast_res_rng])
        axs['B1'].axvline(mod.t0, ls='-', color='grey')
        axs['B1'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
        axs['B1'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
        axs['B1'].set_xlabel(f'Time (MJD)')
        axs['B1'].set_ylabel(f'$\Delta\\alpha \cos \delta$\n(mas)')
        axs['B1'].sharex(axs['F2'])
        axs['B1'].set_title(f'Rolling {time_window} day avg')
        axs['B1'].axhline(0, color='k', ls='--')

        # Astrometry vs. time - North
        axs['B2'].errorbar(t_bin, ast_y_res_bin, yerr=ast_ye_res_bin,
                           ls='none', marker='.')
        axs['B2'].plot(t_mod, ast_resid_mod[:, 1], ls='-', marker='None', zorder=50)
        axs['B2'].set_ylim([-ast_res_rng, ast_res_rng])
        axs['B2'].axvline(mod.t0, ls='-', color='grey')
        axs['B2'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
        axs['B2'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
        axs['B2'].set_xlabel(f'Time (MJD)')
        axs['B2'].set_ylabel(f'$\Delta\delta$\n(mas)')
        axs['B2'].sharex(axs['F2'])
        axs['B2'].axhline(0, color='k', ls='--')

        # Print parameters at the top.
        mod_class = mod.__class__.__name__.split('_')[0]
        title = f'{mod_class:5s}: M_L={mod.mL:.2f} Msun, piRel={mod.piRel:.2f} mas, '
        title += f'thetaE={mod.thetaE_amp:.1f} mas, muRel={mod.muRel_amp:.1f} mas/yr, '
        if 'PSBL' in mod_class:
            title += f'M_L1={mod.mLp:.2f} Msun, M_L2={mod.mLs:.2f} Msun, sepL={mod.sep:.1f} mas, qL={mod.q:.2f} '
        if 'BSPL' in mod_class:
            title += f'sepS={mod.sep:.1f} mas '
            if ~np.isnan(mod.fratio_bin[0]):
                title += f'fratioS={mod.fratio_bin[0]:.1f} '
            else:
                title += f'fratioS=nan '
        if 'BSBL' in mod_class:
            title += f'M_L1={mod.mLp:.2f} Msun, M_L2={mod.mLs:.2f} Msun, sepL={mod.sepL:.1f} mas, qL={mod.q:.2f} '
            title += f'sepS={mod.sepS:.1f} mas '
            title += f'dmagS={mod.mag_src_pri[0]-mod.mag_src_sec[0]:.1f} '
            
                
        axs['AA'].set_title(title, fontsize=16, y=1.05)

        plt.savefig(f'{outdir}/roman_event_lcurves_{filter_name}_{outfile_suffix}.png')
        #plt.close('all')

    # Save parameters to YAML file.
    model_save_file = f'{outdir}/roman_event_model_{filter_name}_{outfile_suffix}.pkl'
    with open(model_save_file, 'wb') as f:
        pickle.dump(mod, f)
        
    # Save the data to an astropy FITS table. We have one for each filter.
    tab.write(f'{outdir}/roman_event_data_{filter_name}_{outfile_suffix}.fits', overwrite=True)
        

    return tab


def plot_roman_lightcurve(mod, tab, t, mod_filt_idx, filter_name,
                          time_window=3, zoom_tE_val=5,
                          outdir='lightcurves/', outfile_suffix=''):
    """
    Plot a Roman lightcurve. 
    """

    ast = np.array([tab[f'x_{filter_name}'], tab[f'y_{filter_name}']]).T
    ast_err = np.array([tab[f'xe_{filter_name}'], tab[f'ye_{filter_name}']]).T
    mag = tab[f'm_{filter_name}']
    mag_err = tab[f'me_{filter_name}']
    
    print(ast.shape)

    # Determine the lensed - unlensed astrometry residuals.
    ast_unlensed = mod.get_astrometry_unlensed(t, filt_idx=mod_filt_idx) * 1e3
    ast_resid = ast - ast_unlensed
    
    # Also make a model curve with 0.5 day time sampling.
    t_mod = np.arange(t.min(), t.max(), 0.5)
    try:
        img_mod, amp_mod = mod.get_all_arrays(t_mod, filt_idx=mod_filt_idx)
        mag_mod = mod.get_photometry(t_mod, filt_idx=mod_filt_idx, amp_arr=amp)
        ast_mod = 1e3 * mod.get_astrometry(t_mod, filt_idx=mod_filt_idx, image_arr=img, amp_arr=amp)

        mag_mod = mag_mod.reshape(len(t_mod))
    except:
        mag_mod = mod.get_photometry(t_mod, filt_idx=mod_filt_idx)
        ast_mod = 1e3 * mod.get_astrometry(t_mod, filt_idx=mod_filt_idx)
    
    ast_unlensed_mod = mod.get_astrometry_unlensed(t_mod, filt_idx=mod_filt_idx) * 1e3
    ast_resid_mod = ast_mod - ast_unlensed_mod
    

    #time_window = 3 # days
    t_bin, ast_x_res_bin, ast_xe_res_bin = moving_average(t, ast_resid[:, 0], time_window, errors=tab[f'xe_{filter_name}'])
    t_bin, ast_y_res_bin, ast_ye_res_bin = moving_average(t, ast_resid[:, 1], time_window, errors=tab[f'ye_{filter_name}'])

    zoom_dt = [mod.t0 - zoom_tE_val*mod.tE, mod.t0 + zoom_tE_val*mod.tE]

    plt_msc = [['F1', 'AA', 'A1'],
               ['F1', 'AA', 'A2'],
               ['F2', 'BB', 'B1'],
               ['F2', 'BB', 'B2']]
    fig, axs = plt.subplot_mosaic(plt_msc,
                                  figsize=(16, 8))
    plt.subplots_adjust(left=0.1, hspace=1.2, wspace=0.5, top=0.92)

    # Photometry vs. time
    axs['F1'].errorbar(t, mag, yerr=mag_err, label=filter_name,
                       ls='none', marker='.', alpha=0.2)
    axs['F1'].plot(t_mod, mag_mod, ls='-', marker='None', zorder=50)
    axs['F1'].set_ylabel(f'{filter_name} mag')
    axs['F1'].axvline(mod.t0, ls='--', color='grey')
    axs['F1'].invert_yaxis()

    # Photometry vs. time -- zoomed
    axs['F2'].errorbar(t, mag, yerr=mag_err, ls='none', marker='.', alpha=0.2)
    axs['F2'].plot(t_mod, mag_mod, ls='-', marker='None', zorder=50)
    axs['F2'].axvline(mod.t0, ls='-', color='grey')
    axs['F2'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
    axs['F2'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
    axs['F2'].set_ylabel(f'Zoomed\n {filter_name} mag')
    axs['F2'].set_xlabel('Time (MJD)')
    axs['F2'].invert_yaxis()
    axs['F2'].set_xlim(zoom_dt)

    # Astrometry on sky
    axs['AA'].errorbar(tab[f'x_{filter_name}'], tab[f'y_{filter_name}'],
                       xerr=tab[f'xe_{filter_name}'], yerr=tab[f'ye_{filter_name}'],
                       ls='none', marker='.', alpha=0.2)
    axs['AA'].plot(ast_mod[:, 0], ast_mod[:, 1], ls='-', marker='None', zorder=50)
    axs['AA'].set_xlabel(f'$\Delta\\alpha \cos \delta$ (mas)')
    axs['AA'].set_ylabel(f'$\Delta\delta$ (mas)')

    # Astrometry vs. time - East
    axs['A1'].errorbar(tab[f't_{filter_name}'], tab[f'x_{filter_name}'], yerr=tab[f'xe_{filter_name}'],
                       ls='none', marker='.', alpha=0.2)
    axs['A1'].plot(t_mod, ast_mod[:, 0], ls='-', marker='None', zorder=50)
    axs['A1'].set_xlabel(f'Time (MJD)')
    axs['A1'].set_ylabel(f'$\Delta\\alpha \cos \delta$\n(mas)')
    axs['A1'].sharex(axs['F1'])

    # Astrometry vs. time - North
    axs['A2'].errorbar(tab[f't_{filter_name}'], tab[f'y_{filter_name}'], yerr=tab[f'ye_{filter_name}'],
                       ls='none', marker='.', alpha=0.2)
    axs['A2'].plot(t_mod, ast_mod[:, 1], ls='-', marker='None', zorder=50)
    axs['A2'].set_xlabel(f'Time (MJD)')
    axs['A2'].set_ylabel(f'$\Delta\delta$\n(mas)')
    axs['A2'].sharex(axs['F1'])


    # Astrometry residuals on sky
    ast_res_rng = np.max([np.std(ast_x_res_bin), np.std(ast_y_res_bin), np.median(ast_xe_res_bin)]) * 3.0
    if ~np.isfinite(ast_res_rng):
        ast_res_rng = 1.0 # mas

    axs['BB'].errorbar(ast_x_res_bin, ast_y_res_bin,
                       xerr=ast_xe_res_bin, yerr=ast_ye_res_bin,
                       ls='none', marker='.', alpha=0.2)
    axs['BB'].plot(ast_resid_mod[:, 0], ast_resid_mod[:, 1], ls='-', marker='None', zorder=50)
    axs['BB'].set_xlabel(f'Lensing Signal\n $\Delta\\alpha \cos \delta$ (mas)')
    axs['BB'].set_ylabel(f'Lensing Signal\n $\Delta\delta$ (mas)')
    axs['BB'].set_xlim([-ast_res_rng, ast_res_rng])
    axs['BB'].set_ylim([-ast_res_rng, ast_res_rng])

    # Astrometry vs. time - East
    axs['B1'].errorbar(t_bin, ast_x_res_bin, yerr=ast_xe_res_bin,
                       ls='none', marker='.')
    axs['B1'].plot(t_mod, ast_resid_mod[:, 0], ls='-', marker='None', zorder=50)
    axs['B1'].set_ylim([-ast_res_rng, ast_res_rng])
    axs['B1'].axvline(mod.t0, ls='-', color='grey')
    axs['B1'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
    axs['B1'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
    axs['B1'].set_xlabel(f'Time (MJD)')
    axs['B1'].set_ylabel(f'$\Delta\\alpha \cos \delta$\n(mas)')
    axs['B1'].sharex(axs['F2'])
    axs['B1'].set_title(f'Rolling {time_window} day avg')
    axs['B1'].axhline(0, color='k', ls='--')

    # Astrometry vs. time - North
    axs['B2'].errorbar(t_bin, ast_y_res_bin, yerr=ast_ye_res_bin,
                       ls='none', marker='.')
    axs['B2'].plot(t_mod, ast_resid_mod[:, 1], ls='-', marker='None', zorder=50)
    axs['B2'].set_ylim([-ast_res_rng, ast_res_rng])
    axs['B2'].axvline(mod.t0, ls='-', color='grey')
    axs['B2'].axvline(mod.t0 - mod.tE, ls='--', color='grey')
    axs['B2'].axvline(mod.t0 + mod.tE, ls='--', color='grey')
    axs['B2'].set_xlabel(f'Time (MJD)')
    axs['B2'].set_ylabel(f'$\Delta\delta$\n(mas)')
    axs['B2'].sharex(axs['F2'])
    axs['B2'].axhline(0, color='k', ls='--')

    # Print parameters at the top.
    mod_class = mod.__class__.__name__.split('_')[0]
    title = f'{mod_class:5s}: M_L={mod.mL:.2f} Msun, piRel={mod.piRel:.2f} mas, '
    title += f'thetaE={mod.thetaE_amp:.1f} mas, muRel={mod.muRel_amp:.1f} mas/yr, '
    if 'PSBL' in mod_class:
        title += f'M_L1={mod.mLp:.2f} Msun, M_L2={mod.mLs:.2f} Msun, sepL={mod.sep:.1f} mas, qL={mod.q:.2f} '
    if 'BSPL' in mod_class:
        title += f'sepS={mod.sep:.1f} mas '
        if ~np.isnan(mod.fratio_bin[0]):
            title += f'fratioS={mod.fratio_bin[0]:.1f} '
        else:
            title += f'fratioS=nan '
    if 'BSBL' in mod_class:
        title += f'M_L1={mod.mLp:.2f} Msun, M_L2={mod.mLs:.2f} Msun, sepL={mod.sepL:.1f} mas, qL={mod.q:.2f} '
        title += f'sepS={mod.sepS:.1f} mas '
        title += f'dmagS={mod.mag_src_pri[0]-mod.mag_src_sec[0]:.1f} '
        
            
    axs['AA'].set_title(title, fontsize=16, y=1.05)

    plt.savefig(f'{outdir}/roman_event_lcurves_{filter_name}_{outfile_suffix}.png')
    
    return


def get_times_roman_gbtds(seasons_fast=(0, 1, 2, 7, 8, 9),
                          seasons_slow=(3, 4, 5, 6),
                          seasons_fast_len=70.5, n_fields_per_set=6,
                          n_sets_f087_fast=1, n_sets_f146_fast=44, dt_gap_fast=0,
                          n_sets_f087_slow=0, n_sets_f146_slow=1, dt_gap_slow=3,
                          t_start = Time('2027-01-01', format='isot', scale='utc'),
                          t_end = Time('2032-01-01', format='isot', scale='utc')):
    """
    Optional
    --------
    seasons_fast : list
        Seasons are spring and fall of each year. But only the seasons_fast
        season indices will be observed at full cadence.
    seasons_slow : list
        Seasons are spring and fall of each year. But the seasons_slow season
        indices will be observed at a slower cadence.
    seasons_fast_len : int
        Number of days in a fast seasons for which we do fast cadence.
        The rest of the time in that seaoson is slow (set by slow cadnece.
    n_fields_per_set : int
        Number of fields to observe.
    n_sets_f087_fast : int
        Number of F087 images to take per set (a set is essentially an
        observing sequence over all the fields in the set).
    n_sets_f149_fast : int
        Number of F149 images to take per set... this is the number of frames
        on a particular field before the whole sequence is repeated.
    dt_gap_fast : float
        Gap (in days) between all the images in a set for all the fields and
        restarting the sequence. During fast seasons, this is typically 0.
    n_sets_f087_slow : int
        Number of F087 images to take per set (a set is essentially an
        observing sequence over all the fields in the set) during slow seasons.
    n_sets_f149_slow : int
        Number of F149 images to take per set... this is the number of frames
        on a particular field before the whole sequence is repeated during
        slow seasons.
    dt_gap_slow : float
        Gap (in days) between all the images in a set for all the fields and
        restarting the sequence. During slow seasons, this is typically
        several days.

    Returns
    -------
    t_f146 : numpy.array
        Times in the F146 filter.
    t_f087 : numpy.array
        Times in the F087 filter.
    """
    # Galactic Center (hopefully will be in GBTDS)
    gc_coord = SkyCoord('17:40:40.04 -29:00:28.0', unit=(u.hourangle, u.deg),
                        obstime='J2000', frame='icrs')

    # Roman launch and survey window of 5 years.
    # t_start =  see input 
    # t_end =  see input

    # First, get coarse daily sampling to figure out Roman visibility windows.
    t_daily = Time(np.arange(t_start.jd, t_end.jd, 1), format='jd')
    time_loc = EarthLocation.of_site('greenwich')

    # get coordinate object for the Sun for each day of the year
    with solar_system_ephemeris.set('builtin'):
        sun_coord = get_body('Sun', t_daily, location=time_loc)

    # Get angular separation of GC LOS to Sun as function of date, in degrees
    sun_angle = sun_coord.separation(gc_coord)

    # allowed angles
    min_sun_angle = (90. - 36.) * u.deg
    max_sun_angle = (90. + 36.) * u.deg

    # Visible days.
    gdx = np.where((sun_angle > min_sun_angle) & (sun_angle < max_sun_angle))[0]

    # Figure out the start of each season,
    # using the time differences of the visible time array, figure out
    dt_vis = np.diff(t_daily[gdx].mjd)
    tdx = np.where(dt_vis > 2)[0]

    t_start_seasons = t_daily[gdx[tdx + 1]].mjd
    t_stop_seasons = t_daily[gdx[tdx]].mjd

    t_start_seasons = Time(np.insert(t_start_seasons, 0,
                                     t_daily[gdx][0].mjd), format='mjd')
    t_stop_seasons = Time(np.insert(t_stop_seasons, len(t_stop_seasons),
                                    t_daily[gdx][-1].mjd), format='mjd')

    # Here is the cycle of observing within the seasons.
    # Fast season:
    dt_f087_fast = (286 * u.s).to(u.d).value  # F087 in fast cadence
    dt_f146_fast = (128 * u.s).to(u.d).value  # W149 at fast cadence

    # Slow season
    dt_f087_slow = (286 * u.s).to(u.d).value  # W149 in slow cadence
    dt_f146_slow = (128 * u.s).to(u.d).value  # W149 in slow cadence

    # Define time arrays that we will fill in. Start with MJD floats.
    t_f146 = np.array([], dtype=float)
    t_f087 = np.array([], dtype=float)

    for ss in range(len(t_start_seasons)):
        t_ss_start_mjd = t_start_seasons[ss].mjd
        t_ss_stop_mjd = t_stop_seasons[ss].mjd

        if ss in seasons_fast:
            if (t_ss_stop_mjd - t_ss_start_mjd) < seasons_fast_len:
                t_ss_stop_mjd = t_ss_start_mjd + seasons_fast_len

            t_cur = t_ss_start_mjd

            # Loop through the cycle until we hit the end of the fast cadence window.
            while t_cur < t_ss_stop_mjd:
                # Start with F087
                ttot_f087_fields = dt_f087_fast * n_fields_per_set
                ttot_f087_fields_sets = ttot_f087_fields * n_sets_f087_fast
                t_f087_cyc = np.arange(t_cur, t_cur + ttot_f087_fields_sets - 1e-5, ttot_f087_fields)
                if len(t_f087_cyc) > 0:
                    t_f087 = np.append(t_f087, t_f087_cyc)
                    t_cur += ttot_f087_fields_sets

                # Now add W149
                ttot_f146_fields = dt_f146_fast * n_fields_per_set
                ttot_f146_fields_sets = ttot_f146_fields * n_sets_f146_fast
                t_f146_cyc = np.arange(t_cur, t_cur + ttot_f146_fields_sets - 1e-5, ttot_f146_fields)
                if len(t_f146_cyc) > 0:
                    t_f146 = np.append(t_f146, t_f146_cyc)
                    t_cur += ttot_f146_fields_sets

                # Now add the gap
                t_cur += dt_gap_fast

            # We are done with fast; but we might have some time left in this season for slow cadence.
            # Rest start/stop so the slow loops below can catch this.
            t_ss_start_mjd = t_ss_stop_mjd
            t_ss_stop_mjd = t_stop_seasons[ss].mjd

        # Time to do all the slow cycles.
        t_cur = t_ss_start_mjd

        # Loop through the cycle until we hit the end of the fast cadence window.
        while t_cur < t_ss_stop_mjd:
            # Start with F087
            ttot_f087_fields = dt_f087_slow * n_fields_per_set
            ttot_f087_fields_sets = ttot_f087_fields * n_sets_f087_slow
            t_f087_cyc = np.arange(t_cur, t_cur + ttot_f087_fields_sets - 1e-5, ttot_f087_fields)
            if len(t_f087_cyc) > 0:
                t_f087 = np.append(t_f087, t_f087_cyc)
                t_cur += ttot_f087_fields_sets

            # Now add W149
            ttot_f146_fields = dt_f146_slow * n_fields_per_set
            ttot_f146_fields_sets = ttot_f146_fields * n_sets_f146_slow
            t_f146_cyc = np.arange(t_cur, t_cur + ttot_f146_fields_sets - 1e-5, ttot_f146_fields)
            if len(t_f146_cyc) > 0:
                t_f146 = np.append(t_f146, t_f146_cyc)
                t_cur += ttot_f146_fields_sets

            # Now add the gap.
            t_cur += dt_gap_slow

    # Test plotting just to visualize.
    # plt.figure(1)
    # plt.clf()
    # f_f146 = np.ones(len(t_f146))
    # f_f087 = np.ones(len(t_f087)) + 0.1
    # plt.plot(t_f146, f_f146, 'k.')
    # plt.plot(t_f087, f_f087, 'r.')
    # #plt.xlim(t_f146.min(), t_f146.min() + 12)
    # plt.ylim(0.9, 1.2)

    return t_f146, t_f087

def run_fisher_analysis_vs_cadence(list_models, t_f146, list_of_tdx, mp_pool_size=1):
    """
    For the list of models, make lightcurves at a range of different cadences
    and run the fisher matrix analysis for each of the cadences.
    Only a subset of fit parameters will be run through the fisher analysis.

    Parameters
    ----------
    list_models : list
        List of BAGLE model instances.
    t_f146 : numpy array (dtype=float)
        Array of time samples to evaluate the lightcurve at.
    list_of_tdx : list of lists
        The F146 time array will be indexed with the indicces indicated in this list.
        This variable should have the shape;

            list_of_tdx = [[indices for cadence 1], [indices for cadence 2], ...]

        This avoids duplicating the time array and only having to generate photometry
        once.
    """
    # Keep some final lists:
    fisher_pnames = []
    fisher_cov_mat = []

    # for ee in range(len(event_tab_t)):
    for ee in range(0, len(list_models)):
        print(f'Starting event {ee}')
        mod = list_models[ee]

        # Make the light curve. This is to get the noise properties.
        print(f'\t Making lightcurve')
        tab = make_roman_lightcurve(mod, t_f146, mod_filt_idx=0, plot=False, noise=False)

        # Set the model parameters around which to estimate the uncertainties.
        # Also designate a few parameters we aren't interestd in.
        # Lastly, set the size of the step to take in calculating the derivatives.
        print(f'\t Setting up parameters for error estimation')
        mod_class = type(mod)
        mod_params = {}
        mod_params_fixed = {}
        mod_params_in_results = {}

        for par in mod.fitter_param_names:
            # Set parameter step sizes for gradients.
            # Override which parameters we will estimate errors for.
            if par.endswith('_E'):
                # Put all positions and proper motions into fixed.
                # Also, don't bother estimating errors.
                mod_params[par] = getattr(mod, par[0:-2])[0]
                mod_params_in_results[par] = False
            elif par.endswith('_N'):
                mod_params[par] = getattr(mod, par[0:-2])[1]
                mod_params_in_results[par] = False
            else:
                # Only put non-position, proper motion variables into fisher calc.
                mod_params[par] = getattr(mod, par)
                mod_params_in_results[par] = True

            # Arbitrarily reset x0 away from zero.
            if par.startswith('xS0'):
                mod_params[par] = 0.001
                mod_params_in_results[par] = False

            if par.startswith('t0'):
                mod_params_in_results[par] = False

            if par.startswith('d'):
                mod_params_in_results[par] = False

        for par in mod.phot_param_names:
            mod_params_fixed[par] = getattr(mod, par)

        for par in mod.fixed_param_names:
            mod_params_fixed[par] = getattr(mod, par)

        # print(mod_params)
        # print(mod_params_in_results)
        # print(mod_params_fixed)
        print('\t N_params in Fisher Matrix: ', np.sum([mod_params_in_results[i] for i in mod_params_in_results]))
        print(f'\t model class = {mod.__class__}')

        param_delta = {'mL': 0.001, 'mLp': 0.001, 'mLs': 0.001,
                       't0': 0.1, 't0_p': 0.1, 'beta': 0.001, 'beta_p': 0.001, 'dL': 0.1, 'dL_dS': 0.001, 'dS': 0.1,
                       'xS0_E': 1e-5, 'xS0_N': 1e-5, 'muL_E': 1e-3, 'muL_N': 1e-3, 'muS_E': 1e-3, 'muS_N': 1e-3,
                       'sep': 1e-3, 'sepS': 1e-3, 'sepL': 1e-3,
                       'alpha': 0.01, 'alphaS': 0.01, 'alphaL': 0.01}

        print(f'\t Calculating covariance matrix')
        t_start = time.time()

        # Only use multi-processing for binary lenses.
        if "BL" in str(mod.__class__):
            pool = mp_pool_size
        else:
            pool = 1

        pnames, cov_mat = sensitivity.fisher_cov_matrix_multi_cadence(t_f146, tab['me_F146'],
                                                                      np.array([tab['xe_F146'], tab['ye_F146']]).T * 1e-3,
                                                                      mod_class, mod_params, mod_params_fixed,
                                                                      mod_params_in_results, list_of_tdx,
                                                                      param_delta=param_delta, mp_pool_size=pool)
        t_stop = time.time()
        print(f'\t Runtime for cov mat = {t_stop - t_start:.4f} sec')

        # err_diag1 = np.diag(cov_mat[0, :, :]) ** 0.5
        # err_diag2 = np.diag(cov_mat[1, :, :]) ** 0.5
        #
        # for pp in range(len(pnames)):
        #     print(f'\t {pnames[pp]} error1 = {err_diag1[pp]:.3e}')
        #     print(f'\t {pnames[pp]} error2 = {err_diag2[pp]:.3e}')

        # Save results
        fisher_pnames += [pnames]
        fisher_cov_mat += [cov_mat]

    return fisher_pnames, fisher_cov_mat


def moving_average(time, value, time_window, errors = None):
    """
    Calculate moving average over a time window for the value array.
    """
    time_window_edges = np.arange(time.min(), time.max()+time_window, time_window)

    inds = np.digitize(time, time_window_edges)
    new_value = []
    new_time = []
    if errors is not None:
        new_error = []
    seen = set()
    for bin_idx in inds:
        if bin_idx not in seen:
            bin_arr = value[inds == bin_idx]
            tim_arr = time[inds == bin_idx]
            if errors is not None:
                err_arr = errors[inds == bin_idx]
            if len(bin_arr) > 0:
                new_value.append(np.mean(bin_arr))
                new_time.append(np.mean(tim_arr))
                if errors is not None:
                    new_error.append(np.sqrt(np.sum(err_arr**2)) / len(err_arr))
                seen.add(bin_idx)

    if errors is not None:
        return new_time, new_value, new_error
    else:
        return new_time, new_value

def fit_simple_model(model_inst, t_f146, plot=False, outfile_suffix=''):
    """
    For a single BAGLE model instance, mock up some
    Roman data, then fit the data to a simple model:

    - constant for magnitude
    - linear proper motion + parallax for x and y positions

    Input
    -----
    model_inst : BAGLE model instance
    t_f146 : list
        List of times in MJD.

    Return
    ------
    chi2_dict : dict
        Dictionary containing the chi^2 values for m, x, y, and the DOF.
    mod_simp_params : dict
        Dictionary of the best-fit model parameters for this
        simple model. Parameter names are 'x0', 'vx', 'x0e', 'vxe', 'pi', etc.
    """
    dat = make_roman_lightcurve(model_inst, t_f146, 
                                mod_filt_idx=0, filter_name='F146',
                                noise=True, verbose=False, plot=plot,
                                outdir='lightcurves/', outfile_suffix=outfile_suffix)
    
    # Mean magnitude -- this is the simplest approach.
    # Could use median or sigma clipping for more robustness. (but slower)
    m_mod = dat['m_F146'].mean()
    m_mod_err = dat['m_F146'].std()

    # Make FlyStar Linear Model and run fit on the data. 
    # Note that FlyStar expects +x to increase to the West. 
    # So flip x data, then flip x output parameters. 
    mm = motion_model.Parallax(model_inst.raL, model_inst.decL, obs='roman')
    t_dyear = Time(dat['t_F146'], format='mjd').decimalyear
    t0_dyear = Time(t_f146.mean(), format='mjd').decimalyear

    mm_par, mm_par_err = mm.run_fit(t_dyear,
                                    -dat['x_F146'], dat['y_F146'], 
                                    dat['xe_F146'], dat['ye_F146'],
                                    t0_dyear)
    
    # Positions in milli-arcseconds.
    x_mod, y_mod = mm.get_pos_at_time(mm_par, [t0_dyear], t_dyear)
    
    # Calculate chi^2 difference between data and the simple model fit.
    chi2_m = np.sum( ( dat['m_F146'] - m_mod)**2 / dat['me_F146']**2 )
    chi2_x = np.sum( (-dat['x_F146'] - x_mod)**2 / dat['xe_F146']**2 )
    chi2_y = np.sum( ( dat['y_F146'] - y_mod)**2 / dat['ye_F146']**2 )
    chi2_dict = {'m': chi2_m, 'x': chi2_x, 'y': chi2_y, 'dof': len(dat)}
    
    # Make final fitted parameters for the simple model. 
    mod_simp_params = {'m0': m_mod, 'm0e': m_mod_err,
                       'x0': mm_par[0], 'x0e': mm_par_err[0],
                       'vx': -mm_par[1], 'vxe': mm_par_err[1],
                       'y0': mm_par[2], 'y0e': mm_par_err[2],
                       'vy': mm_par[3], 'vye': mm_par_err[3],
                       'pi': mm_par[4], 'pie': mm_par_err[4],
                       't0': t0_dyear}
    
    if plot:
        plt.figure()
        plt.errorbar(dat['x_F146'], dat['y_F146'], 
                     xerr=dat['xe_F146'], yerr=dat['ye_F146'], 
                     label='Sim Data',
                     alpha=0.05)
        plt.plot(-x_mod, y_mod, 'k-', label='Simple Model')
        plt.legend()
        plt.xlabel('$\Delta\alpha$ (mas)')
        plt.ylabel('$\Delta\delta \sin \alpha$ (mas)')
        plt.title('Simple Model')

    return chi2_dict, mod_simp_params    

def fit_simple_model_to_events(event_tab, mod_inst_list, t_f146):
    """
    Inputs
    ------
    event_tab : astropy.table
       Table of events from PopSyCLE.
    mod_inst_list : list
       List of BAGLE model instances, usually from
       lightcurves.get_bagle_model_list()

    Output
    ------
    Modifies the input event table to add new columns.
    """

    ####  Non-parallel version
    # results = []
    # for ii in range(len(mod_sm)):
    #     res_ii = fit_parallel_func(ii, mod_inst_list[ii])
    
    #     results.append(res_ii)

    ####. Parallel version
    from multiprocessing import Pool
    import tqdm

    args = [(ii, mod_inst_list[ii], t_f146,
             event_tab['field_id'][ii],
             event_tab['obj_id_L'][ii],
             event_tab['obj_id_S'][ii])
            for ii in range(len(mod_inst_list))]
    
    with Pool(8) as p:
        # Use `map` or `imap` instead of `starmap` if `func` only has 1 argument
        for item in tqdm.tqdm(p.starmap(fit_parallel_func, args), total=len(args)):
            results.append(item)
        
    # with Pool(8) as p:
    #     args = [(ii, mod_inst_list[ii], t_f146) for ii in range(len(mod_inst_list))]
    #     results = tqdm.tqdm( p.imap_unordered(fit_parallel_func, args, chunksize=8) )
        
    res_lists = np.array(results).T
    
    # Add parameters to the event table.
    event_tab['simp_chi2_m'] = res_lists[0]
    event_tab['simp_chi2_x'] = res_lists[1]
    event_tab['simp_chi2_y'] = res_lists[2]
    event_tab['simp_m0']  = res_lists[3]
    event_tab['simp_m0e'] = res_lists[4]
    event_tab['simp_x0']  = res_lists[5]
    event_tab['simp_x0e'] = res_lists[6]
    event_tab['simp_y0']  = res_lists[7]
    event_tab['simp_y0e'] = res_lists[8]
    event_tab['simp_vx']  = res_lists[9]
    event_tab['simp_vxe'] = res_lists[10]
    event_tab['simp_vy']  = res_lists[11]
    event_tab['simp_vye'] = res_lists[12]
    event_tab['simp_pi']  = res_lists[13]
    event_tab['simp_pie'] = res_lists[14]
    event_tab['simp_t0']  = res_lists[15]

    return

    
# First make a fitting function. Need this for parallelization.
def fit_parallel_func(ii, mod, t_f146, field_id, obj_id_L, obj_id_S):
    suffix = f'{field_id}_{obj_id_L:010d}_{obj_id_S:010d}'
    chi2_dic, mod_simp_params = fit_simple_model(mod, t_f146, outfile_suffix=suffix)
    
    results = (chi2_dic['m'], chi2_dic['x'], chi2_dic['y'],
               mod_simp_params['m0'], mod_simp_params['m0e'],
               mod_simp_params['x0'], mod_simp_params['x0e'],
               mod_simp_params['y0'], mod_simp_params['y0e'],
               mod_simp_params['vx'], mod_simp_params['vxe'],
               mod_simp_params['vy'], mod_simp_params['vye'],
               mod_simp_params['pi'], mod_simp_params['pie'],
               mod_simp_params['t0'])
    
    return results


def get_model_param_dicts(mod):
    mod_params = {}
    mod_params_fixed = {}

    params_mod = mod.fitter_param_names + mod.phot_param_names
    params_mod_fix = mod.fixed_param_names
    
    for par in params_mod:
        if par.endswith('_E'):
            mod_params[par] = getattr(mod, par[0:-2])[0]
        elif par.endswith('_N'):
            mod_params[par] = getattr(mod, par[0:-2])[1]
        elif par.startswith('log'):
            mod_params[par] = np.log10(getattr(mod, par.split('_')[1]))
        else:
            mod_params[par] = getattr(mod, par)


    for par in params_mod_fix:
        mod_params_fixed[par] = getattr(mod, par)
           
    return mod_params, mod_params_fixed
    

def get_roman_noise(mag, tint, filter_name):
    # https: // roman.gsfc.nasa.gov / science / WFI_technical.html
    # Zeropoints are the 57 sec point source, 5 sigma.
    zeropoint_all = {'F062': 24.77, 'F087': 24.46, 'F106': 24.46, 'F129': 24.43,
                     'F158': 24.36, 'F184': 23.72, 'F213': 23.14, 'F146': 25.37}
    flux0 = 5**2 * (tint / 57.0)
    fwhm_all = {'F062': 58.0, 'F087': 73.0, 'F106': 87.0, 'F129': 106.0,
                'F158': 128., 'F184': 146., 'F213': 169., 'F146': 105.}

    zp = zeropoint_all[filter_name]  # SNR=5
    fwhm = fwhm_all[filter_name] # mas
    
    ##
    ## Photometric noise vs. mag
    ##
    flux = flux0 * 10 ** ((mag - zp) / -2.5)
    snr = flux ** 0.5
    mag_err = 1.0857 / snr
    mag_err[mag_err < 0.01] = 0.01

    ##
    ## Astrometric noise vs. mag
    ##
    # Assign astrometric errors as FWHM / 2*SNR or 0.1 mas minimum.
    ast_err = fwhm / (8 * snr)  # mas
    ast_err[ast_err < 0.2] = 0.2
    
    return mag_err, ast_err


def plot_noise_model(tint=57, filter_name='F146'):

    # Get a grid of magnitudes
    mag = np.arange(9, 27, 0.1)

    mag_err, ast_err = get_roman_noise(mag, tint, filter_name)

    ##
    ## Plot
    ##
    plt.close('all')
    plt.figure(figsize=(8, 3))
    plt.semilogy(mag, mag_err, 'k.')
    plt.xlabel(f'{filter_name} (mag)')
    plt.ylabel(r'$\sigma_{phot}$ (mag)')

    plt.figure(figsize=(8,3))
    plt.semilogy(mag, ast_err, 'k.')
    plt.xlabel(f'{filter_name} (mag)')
    plt.ylabel(r'$\sigma_{ast}$ (mas)')

    return
    
