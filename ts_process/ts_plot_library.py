#!/usr/bin/env python3
"""
BSD 3-Clause License

Copyright (c) 2018, Southern California Earthquake Center
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Library of functions to plot timeseries
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ts_library import get_points, FAS, calculate_rd50

def plot_overlay_timeseries(args, filenames, stations,
                            output_file, plot_title=None):
    """
    Plotting a comparison of multiple timeseries
    """
    all_styles = ['k', 'r', 'b', 'm', 'g', 'c', 'y', 'brown',
                  'gold', 'blueviolet', 'grey', 'pink']

    # Check number of input timeseries
    if len(stations) > len(all_styles):
        print("[ERROR]: Too many timeseries to plot!")
        sys.exit(-1)

    delta_ts = [station[0].dt for station in stations]
    xtmin = args.xmin
    xtmax = args.xmax
    min_is = [int(xtmin/delta_t) for delta_t in delta_ts]
    max_is = [int(xtmax/delta_t) for delta_t in delta_ts]

    # Create plot
    f, axarr = plt.subplots(nrows=3, ncols=3, figsize=(14, 9))

    # For each component: N/S, E/W, U/D
    for i in range(0, 3):

        signals = [station[i] for station in stations]
        samples = [signal.samples for signal in signals]
        displs = [signal.dis for signal in signals]
        vels = [signal.vel for signal in signals]
        accs = [signal.acc for signal in signals]

        # Get orientation
        orientation = signals[0].orientation
        if type(orientation) is not str:
            orientation = str(int(orientation))

        # Set up titles
        title_acc = "Acceleration : %s" % (orientation)
        title_vel = "Velocity : %s" % (orientation)
        title_dis = "Displacement : %s" % (orientation)

        # cutting signal by bounds
        c_displs = [dis[min_i:max_i] for dis, min_i, max_i in zip(displs,
                                                                  min_is,
                                                                  max_is)]
        c_vels = [vel[min_i:max_i] for vel, min_i, max_i in zip(vels,
                                                                min_is,
                                                                max_is)]
        c_accs = [acc[min_i:max_i] for acc, min_i, max_i in zip(accs,
                                                                min_is,
                                                                max_is)]
        times = [np.arange(xtmin,
                           min(xtmax, (delta_t * sample)),
                           delta_t) for delta_t, sample in zip(delta_ts,
                                                               samples)]

        axarr[i][0] = plt.subplot2grid((3, 3), (i, 0))
        axarr[i][0].set_title(title_dis)
        axarr[i][0].grid(True)
        styles = all_styles[0:len(times)]
        for timeseries, c_dis, style in zip(times, c_displs, styles):
            axarr[i][0].plot(timeseries, c_dis, style)
        plt.xlim(xtmin, xtmax)

        axarr[i][1] = plt.subplot2grid((3, 3), (i, 1))
        axarr[i][1].set_title(title_vel)
        axarr[i][1].grid(True)
        styles = all_styles[0:len(times)]
        for timeseries, c_vel, style in zip(times, c_vels, styles):
            axarr[i][1].plot(timeseries, c_vel, style)
        plt.xlim(xtmin, xtmax)

        axarr[i][2] = plt.subplot2grid((3, 3), (i, 2))
        axarr[i][2].set_title(title_acc)
        axarr[i][2].grid(True)
        styles = all_styles[0:len(times)]
        for timeseries, c_acc, style in zip(times, c_accs, styles):
            axarr[i][2].plot(timeseries, c_acc, style)
        # Add labels to first plot
        if i == 0:
            plt.legend(filenames, prop={'size':6})
        plt.xlim(xtmin, xtmax)

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
        fmt =' png'
    elif output_file.lower().endswith(".pdf"):
        fmt = 'pdf'
    else:
        print("[ERROR]: Unknown format!")
        sys.exit(-1)

    plt.savefig(output_file, format=fmt,
                transparent=False, dpi=300)

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

    rd50s = [calculate_rd50(station,
                            min_i,
                            max_i,
                            tmin,
                            tmax,
                            cut_flag) for station, min_i, max_i in zip(stations,
                                                                       min_is,
                                                                       max_is)]

    f, axarr = plt.subplots(nrows=3, ncols=3, figsize=(14, 9))
    for i in range(0, 3):
        signals = [station[i] for station in stations]
        samples = [signal.samples for signal in signals]
        vels = [signal.vel for signal in signals]
        psas = [psa[i+1] for psa in rd50s]
        periods = [psa[0] for psa in rd50s]
        # Get title
        title = "Velocity component : %s" % (signals[0].orientation)
        if type(title) is not str:
            title = str(int(title))

        for sample, max_i, delta_t in zip(samples, max_is, delta_ts):
            if sample - 1 < max_i:
                print("[ERROR]: t_max has to be under %f" %
                      ((sample - 1) * delta_t))
                sys.exit(1)

        # cutting velocity signal by bounds
        c_vels = [vel[min_i:max_i] for vel, min_i, max_i in zip(vels,
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
        else:
            freqs, fas_s = zip(*[FAS(vel,
                                     delta_t,
                                     points,
                                     xfmin,
                                     xfmax,
                                     3) for vel, delta_t in zip(vels,
                                                                delta_ts)])

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
        axarr[i][2].set_title("PSA(g) versus T(s)")
        axarr[i][2].set_xscale('log')
        axarr[i][2].grid(True)
        for psa, period, style in zip(psas, periods, styles):
            axarr[i][2].plot(period, psa, style)

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
