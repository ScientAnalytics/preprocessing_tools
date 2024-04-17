from os import path
import pathlib

import xarray as xr
import numpy as np
from spectral.io import envi
from scipy import spatial
# import scipy.interpolate as interpolate
from scipy import interpolate

"""This module includes a couple of useful functions
to load hyperspectral data from the ENVI format to a
xarray DataArray.
"""


def load_image(fname):
    """
    Load a hyperspectral image given a file name.

    Parameters
    ----------
    fname : string
        Location of the file

    Returns
    -------
    xarray.DataArray
        The hyperspectral image loaded with its metadata.
    """
    data = load_HSI_image(fname)
    return convert_HSI_to_DataArray(data, fname)


def load_HSI_image(fname):
    """
    Load a hyperspectral image using Spectral Python's envi reader.

    Parameters
    ----------
    fname : string
        File location. Valid file extensions are .hdr, .dat or nothing

    Returns
    -------
    SPy.image
        The loaded image in Spectral Python's numpy-like format
    """
    basename, ext = path.splitext(fname)
    if ext == '.hdr':
        data = envi.open(fname)
    elif ext.lower() == '.dat':
        data = envi.open(basename + '.hdr')
    else:
        try:
            data = envi.open(fname + '.hdr')
        except FileNotFoundError:
            print('Check the filename is correct.')
    return data


def convert_HSI_to_DataArray(data, fname):
    """
    Conerts a Spectral Python image to an xarray.DataArray.
    The latter has much nicer handling for things like wavelength names
    matching their actual values, instead of needing integer indexing,
    built-in plotting and easy selection of regions.

    Parameters
    ----------
    data : SPy.image
        The HSI image to be converted

    Returns
    -------
    xarray.DataArray
        A DataArray representation of the image. The image's metadata will be
        converted to sensible numbers and saved as the attributes in the
        DataArray.
    """
    for k, v in data.metadata.items():
        if k == 'wavelength' or k == 'default bands':
            data.metadata[k] = [float(n) for n in v]
            continue
        try:
            data.metadata[k] = int(v)
        except (ValueError, TypeError):
            pass

    da = xr.DataArray(
        data.asarray(),
        coords={'lines': np.arange(int(data.metadata['lines'])),
                'samples': np.arange(int(data.metadata['samples'])),
                'wavelength': np.array(data.metadata['wavelength']),
                },
        attrs=data.metadata,
    )
    da.attrs['filename'] = fname

    return da


def save_HSI_to_disc(fname, da, metadata=None):
    """
    Save a xarray.DataArray to the given location. This DataArray should be
    one that was created using the load functions above, since that will
    contain the metadata obtained from the .hdr file

    Parameters
    ----------
    fname : string
        The location to save the file to. Should end in .hdr
        If the parent directories for the file do not exist, they will be
        created, so ensure that write permissions are enabled.
    da : xarray.DataArray
        The DataArray to save. Assumes metadata is stored in the DataArray's
        .attrs attribute.
    metadata : Dictionary or None
        Metadata to use when writing image's .hdr. If None is supplied will
        use da.attrs, by default None

    Returns
    -------
    string
        Message indicating success
    """
    if path.exists(path.dirname(fname)) is False:
        pathlib.Path(path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    envi.save_image('test.hdr', da.values, metadata=da.attrs, force=True)
    return f'Successfully saved to {fname}'


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
