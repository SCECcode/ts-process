#!/usr/bin/env python
"""
Library of functions to plot timeseries
"""
from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ts_library import get_periods, get_points, FAS, cal_acc_response

def comparison_plot(args, filenames, stations,
                    output_file, plot_title=None):
    """
    Plot velocity for data and FAS only acceleration for Response
    """
    all_styles = ['k', 'r', 'b', 'm', 'g', 'c', 'y', 'brown',
                  'gold', 'blueviolet', 'grey', 'pink']

    # Check number of input timeseries
    if len(stations) > len(all_styles):
        print("[ERROR]: Too many timeseries to plot!")
        sys.exit(-1)

    delta_ts = [station[0].dt for station in stations]
    files = [os.path.basename(filename) for filename in filenames]

    xtmin = args.xmin
    xtmax = args.xmax
    xfmin = args.xfmin
    xfmax = args.xfmax
    tmin = args.tmin
    tmax = args.tmax
    cut_flag = args.cut

    min_is = [int(xtmin/delta_t) for delta_t in delta_ts]
    max_is = [int(xtmax/delta_t) for delta_t in delta_ts]

    period = get_periods(tmin, tmax)

    f, axarr = plt.subplots(nrows=3, ncols=3, figsize=(14, 9))
    for i in range(0, 3):
        signals = [station[i] for station in stations]
        samples = [signal.samples for signal in signals]
        vels = [signal.vel for signal in signals]
        accs = [signal.acc for signal in signals]
        # Get title
        title = signals[0].orientation
        if type(title) is not str:
            title = str(int(title))

        for sample, max_i, delta_t in zip(samples, max_is, delta_ts):
            if sample - 1 < max_i:
                print("[ERROR]: t_max has to be under %f" %
                      ((sample - 1) * delta_t))
                sys.exit(1)

        # cutting signal by bounds
        c_vels = [vel[min_i:max_i] for vel, min_i, max_i in zip(vels,
                                                                min_is,
                                                                max_is)]
        c_accs = [acc[min_i:max_i] for acc, min_i, max_i in zip(accs,
                                                                min_is,
                                                                max_is)]
        times = [np.arange(xtmin, xtmax, delta_t) for delta_t in delta_ts]
        points = get_points(samples)

        if cut_flag:
            freqs, fas_s = zip(*[FAS(c_vel,
                                     delta_t,
                                     points,
                                     xfmin,
                                     xfmax,
                                     3) for c_vel, delta_t in zip(c_vels,
                                                                  delta_ts)])
            rsps = cal_acc_response(period, c_accs, delta_ts)
        else:
            freqs, fas_s = zip(*[FAS(vel,
                                     delta_t,
                                     points,
                                     xfmin,
                                     xfmax,
                                     3) for vel, delta_t in zip(vels,
                                                                delta_ts)])
            rsps = cal_acc_response(period, accs, delta_ts)

        axarr[i][0] = plt.subplot2grid((3, 4), (i, 0), colspan=2, rowspan=1)
        axarr[i][0].set_title(title)
        axarr[i][0].grid(True)
        styles = all_styles[0:len(times)]
        for timeseries, c_vel, style in zip(times, c_vels, styles):
            axarr[i][0].plot(timeseries, c_vel, style)

        if i == 0:
            plt.legend(files, prop={'size':8})
        plt.xlim(xtmin, xtmax)

        axarr[i][1] = plt.subplot2grid((3, 4), (i, 2), rowspan=1, colspan=1)
        axarr[i][1].set_title('Fourier Amplitude Spectra')
        axarr[i][1].grid(True, which='both')
        axarr[i][1].set_xscale('log')
        axarr[i][1].set_yscale('log')
        for freq, fas, style in zip(freqs, fas_s, styles):
            axarr[i][1].plot(freq, fas, style)

        tmp_xfmin = 0
        if xfmin < 0.5:
            tmp_xfmin = 0
        else:
            tmp_xfmin = xfmin
        plt.xlim(tmp_xfmin, xfmax)

        axarr[i][2] = plt.subplot2grid((3, 4), (i, 3), rowspan=1, colspan=1)
        axarr[i][2].set_title("Response Spectra")
        axarr[i][2].set_xscale('log')
        axarr[i][2].grid(True)
        for rsp, style in zip(rsps, styles):
            axarr[i][2].plot(period, rsp, style)

        plt.xlim(tmin, tmax)

    # Make nice plots with tight_layout
    f.tight_layout()

    # Add overall title if provided
    if plot_title is not None:
        st = plt.suptitle(plot_title, fontsize=16)
        # shift subplots down:
        #st.set_y(0.95)
        f.subplots_adjust(top=0.92)

    # All done, save plot
    if output_file.lower().endswith(".png"):
        fmt = 'png'
    elif output_file.lower().endswith(".pdf"):
        fmt = 'pdf'
    else:
        print("[ERROR]: Unknown format!")
        sys.exit(-1)

    plt.savefig(output_file, format=fmt,
                transparent=False, dpi=300)
# end of comparison_plot