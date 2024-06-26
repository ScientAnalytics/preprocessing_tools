from joblib import Parallel, delayed

from scipy import interpolate, spatial
import numpy as np
import xarray as xr


def continuum_correction_spectrum(spec, spec_wavelength):
    """
    Perform continuum correction on a single spectrum. This entails the
    calculation of the convex hull of the data and then removing that convex
    hull from the original data. This will tend to flatten out the data.

    This implementation also adds two dummy points to be considered. These are
    at the start and end wavelengths, and have value zero. The effect of this
    is to prevent low troughs that fall below a direct line from the start and
    end points from being considered as vertices of the convex hull.

    Parameters
    ----------
    spec : numpy.array
        The values of the spectrogram. Must be the same size as spec_wavelength
    spec_wavelength : numpy.array
        The values of the wavelengths of each reading. Must be the same size
        as spec

    Returns
    -------
    numpy.array
        The continuum-corrected spectrogram
    """
    assert spec_wavelength.ndim == 1, f'ndims of wvl: {spec_wavelength.ndim}'
    assert spec.ndim == 1, f'ndims of spec: {spec.ndim}'
    assert spec.size == spec_wavelength.size, f'spec.size {spec.size} != wvl.size {spec_wavelength.size}'
    # Make a 2D np.array of our data to create a convex hull around
    # This array has spectral values in first column, wavelengths in second
    # We will also add dummy values of 0 to the first and last wavelengths
    # This makes sure that the convex hull does not catch low troughs
    points = np.column_stack((np.append(np.insert(spec, 0, 0), 0),
                              np.append(
                                  np.insert(
                                      spec_wavelength, 0, spec_wavelength[0]),
                                  spec_wavelength[-1])))

    # points = points[~np.isnan(points)]
    points = np.nan_to_num(points, nan=0, posinf=1, neginf=0)

    # we can then calculate the actual convex hull
    try:
        hull = spatial.ConvexHull(points)
    except ValueError:
        print(points)

    # We get back indices, so we need to shift to allow for the extra one that
    # was added at the start of the array. Not doing this leaves us off-by-one
    vertices = np.array(sorted(hull.vertices - 1)[1:-1])

    # Handle the case where we might have a found vertex index bigger than our
    # array. This should happen for the last point.
    vertices = np.where(vertices == len(spec_wavelength), 0, vertices)

    # Get the value of the sprectrum at a given hull point
    try:
        hull_points = spec[vertices]
    except IndexError:
        hull_points = spec[vertices[:-1]]

    wvl = spec_wavelength[vertices]
    # Make our interpolator
    continuum_interpolator = interpolate.interp1d(
        wvl, hull_points,
        bounds_error=False
    )

    # Interpolate our wavelengths
    continuum = continuum_interpolator(spec_wavelength)
    final = np.array(spec / continuum)
    return final


def hull_correction(spec):
    """This only works on a single spectrum, and can not act as the inner
    part of `apply_ufunc`, because that passes numpy arrays while this expects
    to get xarrary.DataArrays. Very frustrating, because this works easily."""
    # Collect the various points that we will use for our hull
    # spatial.ConvexHull expects tuples of points or a 2d numpy
    # array. This provides the latter.
    # We also add a pair of dummy points to the start and end of
    # the array, to avoid catching low values as being outside the hull.
    points = np.column_stack((np.append(np.insert(spec.values, 0, 0), 0),
                              np.append(
                                  np.insert(
                                      spec.wavelength.values, 0, spec.wavelength.values[0]),
                                  spec.wavelength.values[-1])))

    # we can then calculate the actual convex hull
    hull = spatial.ConvexHull(points)

    # we need to shift all the indices, since points has extra elements that spec does not
    vertices = np.array(sorted(hull.vertices - 1))
    vertices = np.pad(vertices, pad_width=(0, len(spec.wavelength.values)-len(hull.vertices)), mode='minimum')
    vertices = np.where(vertices == len(spec.wavelength.values), 0, vertices)
    try:
        hull_points = spec[vertices]
    except IndexError:
        hull_points = spec[vertices[:-1]]

    return _continuum_correction(hull_points, spec)


def _continuum_correction(hull_points, spec):
    continuum = hull_points.drop_duplicates(dim='wavelength')
    continuum_interpolator = interpolate.interp1d(
        x=continuum.wavelength,
        y=continuum,
    )
    continuum = continuum_interpolator(spec.wavelength)
    corrected_spec = spec / continuum

    return corrected_spec


def continuum_correction_image(da):
    """
    Do continuum correction over an entire image.

    This is horrifically slow, and we should look at improving this step in
    the processing toolchain.

    Parameters
    ----------
    da : xarray.DataArray
        The image for which continuum correction is desired

    Returns
    -------
    xarray.DataArray
        The continuum-corrected image
    """
    cont = xr.apply_ufunc(continuum_correction_spectrum,
                          da,
                          da.wavelength,
                          input_core_dims=[['wavelength'], ['wavelength']],
                          output_core_dims=[['wavelength']],
                          vectorize=True
                          )
    cont.attrs = da.attrs
    return cont


def continuum_image(hsi):
    c_corr = np.array(Parallel(n_jobs=-1)(
    delayed(continuum_correction_spectrum)(pixel, hsi.wavelength) for pixel in hsi.values.reshape(-1, hsi.shape[-1])
)).reshape(hsi.shape)
    cont = xr.DataArray(c_corr,
                        coords={'lines': hsi.lines,
                                'samples': hsi.samples,
                                'wavelength': hsi.wavelength
                                }
                        )
    return cont
