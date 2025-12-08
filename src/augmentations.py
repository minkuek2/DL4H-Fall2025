import numpy as np
import scipy.interpolate as interp

def jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma)
    return x * factor

def shift(x, max_shift=5):
    shift_val = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift_val)

def permutation(x, n_segments=4):
    seg_len = len(x) // n_segments
    segments = [x[i*seg_len:(i+1)*seg_len] for i in range(n_segments)]
    np.random.shuffle(segments)
    return np.concatenate(segments)

def magnitude_warp(x, sigma=0.2, knot=4):
    orig = np.linspace(0, 1, len(x))
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    xs = np.linspace(0, 1, knot + 2)
    interp_func = interp.interp1d(xs, random_warps, kind='cubic')
    warping = interp_func(orig)
    return x * warping

def time_warp(x, sigma=0.2, knot=4):
    orig = np.linspace(0, 1, len(x))
    random_curve = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    xs = np.linspace(0, 1, knot + 2)
    interp_func = interp.interp1d(xs, random_curve, kind='cubic')
    tt = interp_func(orig)
    return np.interp(orig, tt, x)
    