from scipy import interpolate, spatial
import numpy as np


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
