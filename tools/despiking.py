from joblib import Parallel, delayed

import numpy as np
import dask
import scipy.signal as ssig


def fix_outlier_args(spectrogram, threshold=0.15):
    fixed_spectrogram = spectrogram.copy()

    # find anomolous troughs
    # we do this by inverting the data and finding the peaks
    lows, dist = ssig.find_peaks(-fixed_spectrogram, threshold=threshold)

    # fix low points
    fixed_lows = fixed_spectrogram[lows] + (dist['left_thresholds'] + dist['right_thresholds']) / 2

    fixed_spectrogram[lows] = fixed_lows

    return np.array(fixed_spectrogram)


def despike_image(hsi, threshold):
    flattened = hsi.values.reshape(-1, hsi.shape[-1])
    # parallel = Parallel(n_jobs=2)
    despiked = np.apply_along_axis(fix_outlier_args, 1, flattened, threshold)
    # output_generator = parallel(delayed(fix_outlier_args)(i ** 2) for i in range(10))
    despiked = despiked.reshape(hsi.shape)

    return despiked
