import pandas as pd
import numpy as np
import re
from scipy.optimize import curve_fit


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