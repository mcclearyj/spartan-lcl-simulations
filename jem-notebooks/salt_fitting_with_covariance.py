"""SALT light curve fitting helpers for LightCurveLynx simulations.

These live in a module (rather than in the notebook) so they can be run in parallel
with a process pool, both in a notebook and in batch jobs on a cluster.
"""

import numpy as np
import sncosmo
from astropy.table import Table

from lightcurvelynx.astro_utils.passbands import PassbandGroup

# LightCurveLynx fluxes are in nJy, which corresponds to an AB zero point of 2.5 * log10(3631 Jy / 1 nJy) = 31.4.
ZP_NJY = 2.5 * np.log10(3631.0e9)

FIT_PARAMS = ["t0", "x0", "x1", "c"]

# Only fit bands whose rest-frame effective wavelength falls in SALT3's well-trained range (Mitra et al. 2025).
REST_WAVE_RANGE = (2800.0, 17000.0)

# Mapping from LightCurveLynx filter name to the registered sncosmo band name.
BAND_NAMES = {}

# Mapping from the registered sncosmo band name to its effective wavelength (in Angstroms).
BAND_WAVE_EFF = {}


def register_passbands(passband_group=None, **kwargs):
    """Register LightCurveLynx's passbands with sncosmo so the fit uses exactly the same
    throughput curves as the simulation.

    The normalized system response is phi(lambda) ∝ T(lambda) / lambda, so multiplying by
    lambda recovers the (trimmed and resampled) throughput up to a constant.

    Parameters
    ----------
    passband_group : PassbandGroup, optional
        The passbands to register. If None, loads the LSST preset (passing along any kwargs).
    """
    if passband_group is None:
        passband_group = PassbandGroup.from_preset(preset="LSST", **kwargs)
    for pb in passband_group:
        waves = pb.normalized_system_response[:, 0]
        trans = pb.normalized_system_response[:, 1] * waves
        band = sncosmo.Bandpass(waves, trans / trans.max(), name=f"lcl_{pb.full_name}")
        sncosmo.register(band, force=True)
        BAND_NAMES[pb.filter_name] = band.name
        BAND_WAVE_EFF[band.name] = band.wave_eff
    return BAND_NAMES


def prepare_data(lightcurve, redshift, model, min_points=5, min_snr=5.0):
    """Build an sncosmo data table from one simulated light curve.

    Uses the *noisy* fluxes, drops saturated or invalid points, and drops bands that are
    not fully covered by the model at this redshift or whose rest-frame effective wavelength
    is outside REST_WAVE_RANGE (Mitra et al. 2025 cut 9).

    Parameters
    ----------
    lightcurve : pandas.DataFrame or None
        The nested light curve for a single object.
    redshift : float
        The redshift of the object.
    model : sncosmo.Model
        The SALT model (used to check band overlap).
    min_points : int
        The minimum number of usable points.
    min_snr : float
        At least one point must have S/N above this value.

    Returns
    -------
    data : astropy.table.Table or None
        The data table, or None if the object does not pass the detection criteria.
    """
    if lightcurve is None or len(lightcurve) < min_points:
        return None

    filters = np.asarray(lightcurve["filter"], dtype=str)
    mjd = np.asarray(lightcurve["mjd"], dtype=float)
    flux = np.asarray(lightcurve["flux"], dtype=float)
    fluxerr = np.asarray(lightcurve["fluxerr"], dtype=float)
    good = np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
    good &= ~np.asarray(lightcurve["is_saturated"], dtype=bool)

    bands = np.array([BAND_NAMES[f] for f in filters])
    model.set(z=redshift)
    for band in np.unique(bands):
        rest_wave = BAND_WAVE_EFF[band] / (1.0 + redshift)
        if not model.bandoverlap(band) or not (REST_WAVE_RANGE[0] < rest_wave < REST_WAVE_RANGE[1]):
            good &= bands != band

    if np.sum(good) < min_points or not np.any(flux[good] / fluxerr[good] > min_snr):
        return None

    return Table(
        {
            "time": mjd[good],
            "band": bands[good],
            "flux": flux[good],
            "fluxerr": fluxerr[good],
            "zp": np.full(np.sum(good), ZP_NJY),
            "zpsys": np.full(np.sum(good), "ab"),
        }
    )


def fit_one(args):
    """Fit a single object with a SALT model, holding the redshift fixed at its true value.

    Parameters
    ----------
    args : tuple
        (idx, lightcurve, params_dict, model_name). Packed as a single tuple so
        the function can be used directly with Executor.map().

    Returns
    -------
    row : dict or None
        The fit results and coverage statistics, or None if the object was not
        detected or the fit failed.
    """
    idx, lightcurve, params_dict, model_name = args
    if not BAND_NAMES:
        register_passbands()

    redshift = params_dict["source.redshift"]
    model = sncosmo.Model(source=model_name)
    data = prepare_data(lightcurve, redshift, model)
    if data is None:
        return None

    try:
        # Stick with uniform priors but shrink range of x1.
        result, fitted_model = sncosmo.fit_lc(
            data,
            model,
            FIT_PARAMS,
            bounds={"t0": (data["time"].min() - 30, data["time"].max() + 30), "x1": (-8, 8), "c": (-0.5, 0.5)},
            phase_range=(-15.0, 45.0),  # Only use data within SALT's well-trained phase range.
            warn=False,
        )
    except (RuntimeError, ValueError):
        return None

    # Record the light curve coverage relative to the *fitted* t0, for the quality cuts.
    fit_phase = (data["time"] - fitted_model["t0"]) / (1.0 + redshift)
    # This is a S/N by the way.
    snr = data["flux"] / data["fluxerr"]
    detected = snr > 3.0

    # The number of bands with maximum S/N > 4 among the points used in the fit (Mitra et al. 2025 cut 1).
    in_fit = (fit_phase > -15.0) & (fit_phase < 45.0)
    n_bands_snr4 = len(np.unique(data["band"][in_fit & (snr > 4.0)]))

    row = {
        "idx": idx,
        "redshift": redshift,
        "success": result.success,
        "ndof": result.ndof,
        "chisq": result.chisq,
        "n_near_peak": np.sum((fit_phase > -10) & (fit_phase < 35)),
        "n_pre_peak": np.sum(detected & (fit_phase < 0) & (fit_phase > -15)),
        "n_post_peak": np.sum(detected & (fit_phase > 5) & (fit_phase < 45)),
        "n_bands_snr4": n_bands_snr4,
    }
    for name in FIT_PARAMS:
        row[f"true_{name}"] = params_dict[f"source.{name}"]
        row[f"fit_{name}"] = result.parameters[result.param_names.index(name)]
        row[f"err_{name}"] = result.errors.get(name, np.nan)

    # sncosmo.fit_lc already computes the covariance of the FITTED parameters.
    # Its axes follow result.vparam_names (NOT result.param_names, which also
    # includes the fixed redshift). Save the ten independent entries of the
    # symmetric 4x4 matrix as scalar columns that are easy to write to Parquet.
    names = list(result.vparam_names)
    cov = result.covariance
    cov = None if cov is None else np.asarray(cov, dtype=float)
    valid_cov = (
        cov is not None and cov.shape == (len(names), len(names))
        and np.all(np.isfinite(cov))
    )
    cov_index = {name: i for i, name in enumerate(names)}
    row["covariance_available"] = bool(valid_cov and all(p in cov_index for p in FIT_PARAMS))

    for i, p in enumerate(FIT_PARAMS):
        for q in FIT_PARAMS[i:]:
            row[f"cov_{p}_{q}"] = (
                float(cov[cov_index[p], cov_index[q]])
                if row["covariance_available"] else np.nan
            )
    return row


def apply_quality_cuts(fit_results):
    """Add columns flagging the objects that pass two sets of light curve quality cuts.

    passes_cuts: the Mitra et al. (2025) selection. Requires a converged fit with at least three
    bands with maximum S/N > 4 (cut 1), |x1| < 3 (cut 2), |c| < 0.3 (cut 3), sigma_x1 < 1 (cut 4),
    sigma_t0 < 2 days (cut 5), and 0.01 < z < 1.4 (cut 8, applied to the true redshift). Cut 9
    is applied when choosing the bands to fit in prepare_data(). Cuts 6 and 7 (photometric
    classification and spectroscopic redshift) do not apply here.

    passes_dp2_cuts: the looser cuts from the DP2 notebook, kept for comparison. Requires a
    converged fit with at least 5 points in [-10, +35] days, a detection before peak, a
    detection after peak, and a reasonable reduced chi-square.
    """
    fit_results["reduced_chisq"] = fit_results["chisq"] / fit_results["ndof"]
    converged = fit_results["success"] & (fit_results["ndof"] > 0)
    fit_results["passes_cuts"] = (
        converged
        & (fit_results["n_bands_snr4"] >= 3)
        & (fit_results["fit_x1"].abs() < 3.0)
        & (fit_results["fit_c"].abs() < 0.3)
        & (fit_results["err_x1"] < 1.0)
        & (fit_results["err_t0"] < 2.0)
        & (fit_results["redshift"] > 0.01)
        & (fit_results["redshift"] < 1.4)
    )
    fit_results["passes_dp2_cuts"] = (
        converged
        & (fit_results["n_near_peak"] >= 5)
        & (fit_results["n_pre_peak"] >= 1)
        & (fit_results["n_post_peak"] >= 1)
        & (fit_results["reduced_chisq"] < 3.0)
    )
    return fit_results
