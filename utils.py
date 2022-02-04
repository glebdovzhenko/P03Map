import pandas as pd
import numpy as np
import re
from scipy.optimize import curve_fit
from typing import Union, Dict


def read_fio(f_name):
    header = dict()
    data = pd.DataFrame()

    param_line = re.compile(r'^(?P<key>[\w\.]+) = (?P<val>[\d\.+-eE]+)\n')
    t_header_line = re.compile(r'^ Col (?P<col>[\d]+) (?P<key>[\w\.]+) (?P<type>[\w\.]+)\n')

    with open(f_name, 'r') as f:
        lines = f.readlines()

        for line in lines[lines.index('%p\n') + 1:]:
            m = param_line.match(line)
            if m:
                header[m.group('key')] = float(m.group('val'))

        if not header:
            return header, data

        columns = dict()
        for ii, line in enumerate(lines[lines.index('%d\n') + 1:]):
            m = t_header_line.match(line)
            if m:
                columns[int(m.group('col'))] = m.group('key')
            else:
                break

        if not columns:
            return header, data

        data = pd.DataFrame(columns=list(columns.values()))
        t_row_line = re.compile(r'^' + r'\s+([\w\.+-]+)' * len(columns) + r'\n')

        def _float(s):
            try:
                return float(s)
            except ValueError:
                return None

        for line in lines[lines.index('%d\n') + ii + 1:]:
            m = t_row_line.match(line)
            if m is not None:
                vals = m.groups()
                row = {columns[i + 1]: _float(vals[i]) for i in range(len(columns))}
                data.loc[data.shape[0]] = row

    return header, data


def n_integrate(y_vals, x_vals):
    return np.sum(0.5 * (y_vals[1:] + y_vals[:-1]) * (x_vals[1:] - x_vals[:-1]))


def roonizi_estimate(y_vals, x_vals, nn=4):
    phi_shape = (nn + 4, nn + 4)
    phi = np.zeros(phi_shape)
    f = np.zeros(phi_shape[0])

    phi_funcs = [x_vals ** ii for ii in range(nn + 2)] + \
                [np.array([n_integrate((y_vals * x_vals)[:ii], x_vals[:ii]) for ii in range(x_vals.shape[0])])] + \
                [np.array([n_integrate(y_vals[:ii], x_vals[:ii]) for ii in range(x_vals.shape[0])])]

    for ii in range(phi_shape[0]):
        f[ii] = n_integrate(phi_funcs[ii] * y_vals, x_vals)
        for jj in range(phi_shape[1]):
            phi[ii, jj] = n_integrate(phi_funcs[ii] * phi_funcs[jj], x_vals)

    beta = np.linalg.inv(phi).dot(f.T)

    est_gauss = dict()
    est_gauss['c'] = -beta[nn + 3] / beta[nn + 2]
    est_gauss['sigma'] = np.sqrt(-1. / beta[nn + 2])

    est_poly = np.zeros(nn)
    est_poly[nn - 1] = (nn + 1) * beta[nn + 1] * est_gauss['sigma'] ** 2
    est_poly[nn - 2] = est_gauss['c'] * est_poly[nn - 1] + nn * beta[nn] * est_gauss['sigma'] ** 2
    for k in range(nn - 1, 1, -1):
        est_poly[k - 2] = k * (beta[k] - est_poly[k]) * est_gauss['sigma'] ** 2 + est_gauss['c'] * est_poly[k - 1]
    est_poly = est_poly[::-1]

    est_gauss['a'] = n_integrate(
        (y_vals - np.polyval(est_poly, x_vals)) * np.exp(
            -(x_vals - est_gauss['c']) ** 2 / (2. * est_gauss['sigma'] ** 2.)),
        x_vals) / n_integrate(np.exp(-(x_vals - est_gauss['c']) ** 2 / (est_gauss['sigma'] ** 2.)), x_vals)

    return est_poly, est_gauss


def refine_estimate(xdata, ydata, est_poly, est_gauss):
    def ff(x, *args):
        if np.any(np.polyval(args[3:], x) < 0.):
            return 0. * x

        return np.exp(-(x - args[0]) ** 2 / (2. * args[1] ** 2.)) * args[2] + \
               np.polyval(args[3:], x)

    p0 = np.array([
        est_gauss['c'],
        est_gauss['sigma'],
        est_gauss['a'],
        *est_poly
    ])

    popt, pcov = curve_fit(
        ff, xdata, ydata, p0,
        bounds=[
            (np.min(xdata), 0.,                            0.,     -np.inf, -np.inf),
            (np.max(xdata), np.max(xdata) - np.min(xdata), np.inf, np.inf,   np.inf)
        ]
    )

    est_gauss_ = dict()
    est_gauss_['c'] = popt[0]
    est_gauss_['sigma'] = popt[1]
    est_gauss_['a'] = popt[2]
    est_poly_ = popt[3:]

    return est_poly_, est_gauss_


def en_wl(en: Union[float, np.ndarray, None] = None,
          wl: Union[float, np.ndarray, None] = None
          ) -> Dict[str, Union[float, np.ndarray]]:
    """
    Converts between photon energy in keV and wavelength in AA
    :param en: [keV]
    :param wl: [AA]
    :return: dictionary with keys 'en', 'wl'.
    """
    if en is not None and wl is None:
        return {'en': en, 'wl': 12.39842 / en}
    elif wl is not None and en is None:
        return {'wl': wl, 'en': 12.39842 / wl}
    else:
        raise ValueError('Input kwargs are wl or en.')


def bragg(en: Union[float, np.ndarray, None] = None,
          wl: Union[float, np.ndarray, None] = None,
          k: Union[float, np.ndarray, None] = None,
          tth: Union[float, np.ndarray, None] = None,
          d: Union[float, np.ndarray, None] = None,
          q: Union[float, np.ndarray, None] = None
          ) -> Dict[str, Union[float, np.ndarray]]:
    """
    :param q: inverse lattice parameter [AA^-1]
    :param d: lattice parameter [AA]
    :param tth: 2Theta Bragg angle [deg]
    :param k: photon scattering vector [AA^-1]
    :param wl: photon wavelength [AA]
    :param en: photon energy [keV]
    :return:
    """
    if sum(x is not None for x in [en, wl, k, tth, d, q]) != 2:
        raise ValueError('Too many parameters specified')
    elif sum(x is not None for x in [en, wl, k]) > 1:
        raise ValueError('Too many photon parameters specified')
    elif sum(x is not None for x in [d, q]) > 1:
        raise ValueError('Too many lattice parameters specified')

    if sum(x is not None for x in [en, wl, k]) == 1:
        if k is None:
            tmp = en_wl(en=en, wl=wl)
            en = tmp['en']
            wl = tmp['wl']
            k = 2. * np.pi / wl
        else:
            wl = 2. * np.pi / k
            en = en_wl(wl=wl)['en']
    else:
        if q is not None:
            d = 2. * np.pi / q
        else:
            q = 2. * np.pi / d

        wl = 2. * d * np.sin(np.pi * tth / 360.)
        en = en_wl(wl=wl)['en']
        k = 2. * np.pi / wl

        return {'en': en, 'wl': wl, 'k': k, 'tth': tth, 'd': d, 'q': q}

    if sum(x is not None for x in [d, q]) == 1:
        if q is not None:
            d = 2. * np.pi / q
        else:
            q = 2. * np.pi / d
    else:
        d = wl / (2. * np.sin(np.pi * tth / 360.))
        q = 2. * np.pi / d

        return {'en': en, 'wl': wl, 'k': k, 'tth': tth, 'd': d, 'q': q}

    tth = 360. * np.arcsin(wl / (2. * d)) / np.pi
    return {'en': en, 'wl': wl, 'k': k, 'tth': tth, 'd': d, 'q': q}
