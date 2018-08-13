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

Library of functions to process timeseries
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import math
import atexit
import shutil
import tempfile
import numpy as np
import subprocess
from scipy import interpolate
from scipy.signal import sosfiltfilt, filtfilt, ellip, butter, kaiser
from scipy.integrate import cumtrapz
import matplotlib as mpl
if mpl.get_backend() != 'agg':
    mpl.use('Agg') # Disables use of Tk/X11
import matplotlib.pyplot as plt
import pylab

# This is used to convert from accel in g to accel in cm/s/s
G2CMSS = 980.665 # Convert g to cm/s/s

def cleanup(dir_name):
    """
    This function removes the temporary directory
    """
    shutil.rmtree(dir_name)

class TimeseriesComponent(object):
    """
    This class implements attributes related to a single
    component timeseries, including displacement, velocity,
    and acceleration.
    """
    def __init__(self, samples, dt,
                 orientation,
                 acc, vel, dis):
        """
        Initialize the class attributes with the parameters
        provided by the user
        """
        self.samples = samples
        self.dt = dt
        self.orientation = orientation
        self.acc = acc
        self.vel = vel
        self.dis = dis

def integrate(data, dt):
    """
    compute derivative of a numpy array
    initial condition assumed 0
    result has same size as input
    """
    newdata = cumtrapz(data, dx=dt, initial=0) + data[0] * dt / 2.0
    return newdata
#end integrate

def derivative(data, dt):
    """
    compute derivative of an numpy array
    """
    newdata = np.insert(data, 0, 0)
    newdata = np.diff(newdata) / dt
    return newdata
#end derivative

def calculate_distance(epicenter, st_loc):
    """
    Calculates the distance between two pairs of lat, long coordinates
    using the Haversine formula
    """
    lat1 = math.radians(abs(epicenter[0]))
    lon1 = math.radians(abs(epicenter[1]))
    lat2 = math.radians(abs(st_loc[0]))
    lon2 = math.radians(abs(st_loc[1]))

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371.0

    return c * r
#end calculate_distance

def get_periods(tmin, tmax):
    """
    Return an array of period T
    """
    # tmin = 1/fmax
    # tmax = 1/fmin
    a = np.log10(tmin)
    b = np.log10(tmax)

    periods = np.linspace(a, b, 20)
    periods = np.power(10, periods)

    return periods
#end get_periods

def write_peer_acc_file(peer_fn, acc_ts, delta_t):
    """
    Write acc timeseries into a peer-format file
    """
    # Number of header lines needed in PEER file
    PEER_HEADER_LINES = 6

    # Adjust the header lines needed in the PEER format
    header_lines = []
    while len(header_lines) <= (PEER_HEADER_LINES - 2):
        header_lines.append("\n")

    output_file = open(peer_fn, 'w')
    # Write header
    for line in header_lines[0:(PEER_HEADER_LINES - 2)]:
        output_file.write(line)
    output_file.write("Acceleration in g\n")
    output_file.write("  %d   %1.6f   NPTS, DT\n" %
                      (len(acc_ts), delta_t))
    for index, elem in enumerate(acc_ts):
        output_file.write("% 12.7E " % (elem / G2CMSS))
        if (index % 5) == 4:
            output_file.write("\n")
    output_file.write("\n")
    output_file.close()

def run_rotd50(workdir,
               peer_input_1_file, peer_input_2_file,
               output_rotd50_file):
    """
    Runs the RotD50 code using the inputs provided
    """
    # Make sure we don't have absolute path names
    peer_input_1_file = os.path.basename(peer_input_1_file)
    peer_input_2_file = os.path.basename(peer_input_2_file)
    output_rotd50_file = os.path.basename(output_rotd50_file)
    logfile = "rotd50.log"

    bin_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    # Save cwd, change back to it at the end
    old_cwd = os.getcwd()
    os.chdir(workdir)

    # Make sure we remove the output files first or Fortran will
    # complain if they already exist
    try:
        os.unlink(output_rotd50_file)
    except OSError:
        pass

    #
    # write config file for rotd50 program
    rd50_conf = open("rotd50_inp.cfg", 'w')
    # This flag indicates inputs acceleration
    rd50_conf.write("2 interp flag\n")
    # This flag indicate we are processing two input files
    rd50_conf.write("1 Npairs\n")
    # Number of headers in the file
    rd50_conf.write("6 Nhead\n")
    rd50_conf.write("%s\n" % peer_input_1_file)
    rd50_conf.write("%s\n" % peer_input_2_file)
    rd50_conf.write("%s\n" % output_rotd50_file)
    # Close file
    rd50_conf.close()

    progstring = ("%s >> %s 2>&1" %
                  (os.path.join(bin_dir, "rotd50", "rotd50"), logfile))
    try:
        proc = subprocess.Popen(progstring, shell=True)
        proc.wait()
    except KeyboardInterrupt:
        print("Interrupted!")
        sys.exit(1)
    except:
        print("Unexpected error returned from Subprocess call: ",
              sys.exc_info()[0])

    # Restore working directory
    os.chdir(old_cwd)

def read_rd50(input_rd50_file):
    """
    Reads RotD50 input file
    """
    periods = np.empty(0)
    comp1 = np.empty(0)
    comp2 = np.empty(0)
    input_file = open(input_rd50_file, 'r')
    for line in input_file:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        items = line.split()
        items = [float(item) for item in items]
        periods = np.append(periods, items[0])
        comp1 = np.append(comp1, items[1])
        comp2 = np.append(comp2, items[2])
    input_file.close()

    return periods, comp1, comp2

def calculate_rd50(station, min_i, max_i, tmin, tmax, cut_flag=False):
    """
    Calculates the RotD50 for a given station, if cut_flag is TRUE,
    trims the acc timeseries using min_i and max_i, returns data
    for periods within tmin, tmax
    """
    comp_h1 = station[0].acc
    comp_h2 = station[1].acc
    comp_v = station[2].acc
    delta_ts = [item.dt for item in station]
    peer_fns = ["hor1.acc.peer", "hor2.acc.peer", "ver.acc.peer"]
    rotd50_h_file = "station_h.rd50"
    rotd50_v_file = "station_v.rd50"

    # Trim timeseries if needed
    if cut_flag:
        comp_h1 = comp_h1[min_i:max_i]
        comp_h2 = comp_h2[min_i:max_i]
        comp_v = comp_v[min_i:max_i]

    # Create temp Directory
    temp_dir = tempfile.mkdtemp()
    # And clean up later
    atexit.register(cleanup, temp_dir)

    # Write PEER tempfiles
    for peer_fn, comp, delta_t in zip(peer_fns,
                                      [comp_h1,
                                       comp_h2,
                                       comp_v],
                                      delta_ts):
        write_peer_acc_file(os.path.join(temp_dir, peer_fn),
                            comp, delta_t)

    # Calculate RotD50 outputs
    run_rotd50(temp_dir, peer_fns[0], peer_fns[1], rotd50_h_file)
    run_rotd50(temp_dir, peer_fns[2], peer_fns[2], rotd50_v_file)

    periods, comp1_rd50, comp2_rd50 = read_rd50(os.path.join(temp_dir,
                                                             rotd50_h_file))
    _, compv_rd50, _ = read_rd50(os.path.join(temp_dir, rotd50_v_file))

    # Find only periods we want
    try:
        idx_min = np.nonzero(periods-tmin >= 0)[0][0]
    except:
        idx_min = 0
    try:
        idx_max = np.nonzero(periods-tmax > 0)[0][0]
    except:
        idx_max = len(periods)

    periods = periods[idx_min:idx_max]
    comp1_rd50 = comp1_rd50[idx_min:idx_max]
    comp2_rd50 = comp2_rd50[idx_min:idx_max]
    compv_rd50 = compv_rd50[idx_min:idx_max]

    return [periods, comp1_rd50, comp2_rd50, compv_rd50]

def get_points(samples):
    # points is the least base-2 number that is greater than max samples
    power = int(math.log(max(samples), 2)) + 1
    return 2**power
# end of get_points

def smooth(data, factor):
    # factor = 3; c = 0.5, 0.25, 0.25
    # TODO: fix coefficients for factors other than 3
    c = 0.5 / (factor-1)
    for i in range(1, data.size-1):
        data[i] = 0.5 * data[i] + c * data[i-1] + c * data[i+1]
    return data
#end smooth

def FAS(data, dt, points, fmin, fmax, s_factor):
    afs = abs(np.fft.fft(data, points)) * dt
    freq = (1/dt)*np.array(range(points))/points

    deltaf = (1/dt)/points

    inif = int(fmin/deltaf)
    endf = int(fmax/deltaf) + 1

    afs = afs[inif:endf]
    afs = smooth(afs, s_factor)
    freq = freq[inif:endf]
    return freq, afs
#end FAS

def taper(flag, m, samples):
    # m = samples for taper
    # samples = total samples
    window = kaiser(2*m+1, beta=14)

    if flag == 'front':
        # cut and replace the second half of window with 1s
        ones = np.ones(samples-m-1)
        window = window[0:(m+1)]
        window = np.concatenate([window, ones])

    elif flag == 'end':
        # cut and replace the first half of window with 1s
        ones = np.ones(samples-m-1)
        window = window[(m+1):]
        window = np.concatenate([ones, window])

    elif flag == 'all':
        ones = np.ones(samples-2*m-1)
        window = np.concatenate([window[0:(m+1)], ones, window[(m+1):]])

    # avoid concatenate error
    if window.size < samples:
        window = np.append(window, 1)

    if window.size != samples:
        print(window.size)
        print(samples)
        print("[ERROR]: taper and data do not have the same number of samples.")
        window = np.ones(samples)

    return window
#end taper

def seism_appendzeros(flag, t_diff, m, timeseries):
    """
    dds zeros in the front and/or at the end of an numpy array
    apply taper before adding
    """
    num = int(t_diff / timeseries.dt)
    zeros = np.zeros(num)

    if flag == 'front':
        # applying taper in the front
        if m != 0:
            window = taper('front', m, timeseries.samples)
            timeseries.acc = timeseries.acc * window
            timeseries.vel = timeseries.vel * window
            timeseries.dis = timeseries.dis * window

        # adding zeros in front of data
        timeseries.acc = np.append(zeros, timeseries.acc)
        timeseries.vel = np.append(zeros, timeseries.vel)
        timeseries.dis = np.append(zeros, timeseries.dis)

    elif flag == 'end':
        if m != 0:
            # applying taper in the front
            window = taper('end', m, timeseries.samples)
            timeseries.acc = timeseries.acc * window
            timeseries.vel = timeseries.vel * window
            timeseries.dis = timeseries.dis * window

        timeseries.acc = np.append(timeseries.acc, zeros)
        timeseries.vel = np.append(timeseries.vel, zeros)
        timeseries.dis = np.append(timeseries.dis, zeros)

    timeseries.samples += num

    return timeseries
# end of seism_appendzeros

def seism_cutting(flag, t_diff, m, timeseries):
    """
    cut data in the front or at the end of an numpy array
    apply taper after cutting
    """
    num = int(t_diff / timeseries.dt)
    if num >= timeseries.samples:
        print("[ERROR]: fail to cut timeseries.")
        return timeseries

    if flag == 'front' and num != 0:
        # cutting timeseries
        timeseries.acc = timeseries.acc[num:]
        timeseries.vel = timeseries.vel[num:]
        timeseries.dis = timeseries.dis[num:]
        timeseries.samples -= num

        # applying taper at the front
        window = taper('front', m, timeseries.samples)
        timeseries.acc = timeseries.acc * window
        timeseries.vel = timeseries.vel * window
        timeseries.dis = timeseries.dis * window

    elif flag == 'end' and num != 0:
        num *= -1
        # cutting timeseries
        timeseries.acc = timeseries.acc[:num]
        timeseries.vel = timeseries.vel[:num]
        timeseries.dis = timeseries.dis[:num]
        timeseries.samples += num

        # applying taper at the end
        window = taper('end', m, timeseries.samples)
        timeseries.acc = timeseries.acc * window
        timeseries.vel = timeseries.vel * window
        timeseries.dis = timeseries.dis * window

    return timeseries
# end of seism_cutting

def polimod(x, y, n, m):
    """
    Polymod Fit polynomial to data - by J. Stewart 5/25/98

    polymod(x,y,n,m) finds the coefficients of a polynomial P(x) of
    degree n that fits the data. P(X(I))~=Y(I), in a least-squares sense
    but the first m terms of the polynomial are neglected in forming
    the fit (e.g. to use the squared and cubic terms in a third order
    polynomial, which has 4 coefficients, use n=3, m=1)

    The regression problem is formulated in matrix format as:

    y = G*m or

          3  2
    Y = [x  x  x  1] [p3
                      p2
                      p1
                      p0]

    where the vector p contains the coefficients to be found. For a
    3th order polynomial, matrix G would be:

    G = [x^3 x^2 X^1 1]

    where the number of rows in G equals the number of rows in x.
    """
    # Make sure the 2 vectors have the same size
    if len(x) != len(y):
        print("ERROR: X and Y vectors must be of same size!")
        sys.exit(-1)

    G = np.zeros((len(x), (n-m)))
    for i in range(0, len(x)):
        for j in range(0, (n-m)):
            G[i][j] = x[i] ** (j+1+m)

    # Transpose G
    GT = G.transpose()
    # Form solution see Section 2.2.2 of Geophysics 104 notes
    p = np.dot(np.dot(np.linalg.inv(np.dot(GT, G)), GT), y)
    # Polynomial coefficients are row vectors by convention
    return p

def baseline_function(acc, dt, gscale, ordern):
    """
    Integrates acceleration record and baseline corrects velocity and
    displacement time series using 5th order polynomial without the constant
    and linear term (only square, cubic, 4th and 5th order terms, so that
    the leading constants are applied to disp, vel, and acc)
    """
    # Use gscale to convert to cm/sec2
    acc = acc * gscale
    times = np.linspace(0, (len(acc) - 1) * dt, len(acc))

    # Integrate to get velocity and displacement
    vel = np.zeros(len(acc))
    dis = np.zeros(len(acc))

    vel[0] = (acc[0]/2.0) * dt
    for i in range(1, len(acc)):
        vel[i] = vel[i-1] + (((acc[i-1] + acc[i]) / 2.0) * dt)
    dis[0] = (vel[0]/2.0) * dt
    for i in range(1, len(vel)):
        dis[i] = dis[i-1] + (((vel[i-1] + vel[i]) / 2.0) * dt)

    if ordern == 10:
        p = polimod(times, dis, 10, 1)
        pd = [p[8], p[7], p[6], p[5], p[4], p[3], p[2], p[1], p[0], 0.0, 0.0]
        pv = [10*p[8], 9*p[7], 8*p[6], 7*p[5], 6*p[4], 5*p[3], 4*p[2],
              3*p[1], 2*p[0], 0.0]
        pa = [10*9*p[8], 9*8*p[7], 8*7*p[6], 7*6*p[5], 6*5*p[4],
              5*4*p[3], 4*3*p[2], 3*2*p[1], 2*1*p[0]]
    elif ordern == 5:
        p = polimod(times, dis, 5, 1)
        pd = [p[3], p[2], p[1], p[0], 0.0, 0.0]
        pv = [5*p[3], 4*p[2], 3*p[1], 2*p[0], 0.0]
        pa = [5*4*p[3], 4*3*p[2], 3*2*p[1], 2*1*p[0]]
    elif ordern == 3:
        p = polimod(times, dis, 3, 1)
        pd = [p[1], p[0], 0.0, 0.0]
        pv = [3*p[1], 2*p[0], 0.0]
        pa = [3*2*p[1], 2*1*p[0]]
    else:
        print("ERROR: Baseline function use order 3, 5, or 10!")
        sys.exit(-1)

    # Evalutate polynomial correction at each time step
    dcor = np.polyval(pd, times)
    vcor = np.polyval(pv, times)
    acor = np.polyval(pa, times)

    # Calculate corrected timeseries
    dmod = dis - dcor
    vmod = vel - vcor
    amod = acc - acor

    amod = amod / gscale

    return times, amod, vmod, dmod
# end of baseline_function

def rotate_timeseries(station, rotation_angle):
    """
    The function rotates timeseries for a specific station
    """
    # Check rotation angle
    if rotation_angle is None:
        # Nothing to do!
        return station

    if rotation_angle < 0 or rotation_angle > 360:
        print("[ERROR]: Invalid rotation angle: %f" % (rotation_angle))
        return False

    # Make sure channels are ordered properly
    if station[0].orientation > station[1].orientation:
        # Swap channels
        temp = station[0]
        station[0] = station[1]
        station[1] = temp

    # Figure out how to rotate
    x = station[0].orientation
    y = station[1].orientation

    # Calculate angle between two components
    angle = y - x
    # print("Angle = %d" % (angle))

    # We need two orthogonal channels
    if abs(angle) != 90 and abs(angle) != 270:
        return False

    # Create rotation matrix
    if angle == 90:
        matrix = np.array([(math.cos(math.radians(rotation_angle)),
                            -math.sin(math.radians(rotation_angle))),
                           (math.sin(math.radians(rotation_angle)),
                            math.cos(math.radians(rotation_angle)))])
    else:
        # Angle is 270!
        matrix = np.array([(math.cos(math.radians(rotation_angle)),
                            +math.sin(math.radians(rotation_angle))),
                           (math.sin(math.radians(rotation_angle)),
                            -math.cos(math.radians(rotation_angle)))])

    # Make sure they all have the same number of points
    if len(station[0].acc) != len(station[1].acc):
        n_points = min(len(station[0].acc), len(station[1].acc))
        station[0].acc = station[0].acc[0:n_points-1]
        station[1].acc = station[1].acc[0:n_points-1]
    if len(station[0].vel) != len(station[1].vel):
        n_points = min(len(station[0].vel), len(station[1].vel))
        station[0].vel = station[0].vel[0:n_points-1]
        station[1].vel = station[1].vel[0:n_points-1]
    if len(station[0].dis) != len(station[1].dis):
        n_points = min(len(station[0].dis), len(station[1].dis))
        station[0].dis = station[0].dis[0:n_points-1]
        station[1].dis = station[1].dis[0:n_points-1]

    # Rotate
    [station[0].acc,
     station[1].acc] = matrix.dot([station[0].acc,
                                   station[1].acc])

    [station[0].vel,
     station[1].vel] = matrix.dot([station[0].vel,
                                   station[1].vel])
    [station[0].dis,
     station[1].dis] = matrix.dot([station[0].dis,
                                   station[1].dis])

    # Adjust station orientation after rotation is completed
    station[0].orientation = station[0].orientation - rotation_angle
    station[1].orientation = station[1].orientation - rotation_angle
    if station[0].orientation < 0:
        station[0].orientation = 360 + station[0].orientation
    if station[1].orientation < 0:
        station[1].orientation = 360 + station[1].orientation

    return station
# end of rotate

def filter_timeseries(timeseries, family, btype,
                      N=5, rp=0.1, rs=100,
                      fmin=0.0, fmax=0.0, Wn=None,
                      debug=False):
    """
    Function that filters acc/vel/dis of a timeseries component
    """
    if debug:
        print("[INFO]: Filtering timeseries: %s - %s - fmin=%.2f, fmax=%.2f" %
              (family, btype, fmin, fmax))

    timeseries.acc = filter_data(timeseries.acc, timeseries.dt,
                                 btype=btype, family=family,
                                 fmin=fmin, fmax=fmax,
                                 N=N, rp=rp, rs=rs, Wn=Wn)
    timeseries.vel = filter_data(timeseries.vel, timeseries.dt,
                                 btype=btype, family=family,
                                 fmin=fmin, fmax=fmax,
                                 N=N, rp=rp, rs=rs, Wn=Wn)
    timeseries.dis = filter_data(timeseries.dis, timeseries.dt,
                                 btype=btype, family=family,
                                 fmin=fmin, fmax=fmax,
                                 N=N, rp=rp, rs=rs, Wn=Wn)

    return timeseries

def filter_data(data, dt, family, btype,
                N=5, rp=0.1, rs=100,
                fmin=0.0, fmax=0.0, Wn=None):
    """
    Function that filters timeseries in data
    """
    # Make sure we have a numpy array in the input
    if not isinstance(data, np.ndarray):
        print("[ERROR]: data input for filter is not an numpy array.")
        return data

    # Set up some values
    if Wn is None:
        Wn = 0.05/((1.0/dt)/2.0)
    a = np.array([], float)
    b = np.array([], float)
    w_min = fmin/((1.0/dt)/2.0)
    w_max = fmax/((1.0/dt)/2.0)

    if fmin and fmax and btype == 'bandpass':
        Wn = [w_min, w_max]
    elif fmax and btype == 'lowpass':
        Wn = w_max
    elif fmin and btype == 'highpass':
        Wn = w_min

    # Calling filter
    if family == 'ellip':
        b, a = ellip(N=N, rp=rp, rs=rs, Wn=Wn, btype=btype, analog=False)
        data = filtfilt(b, a, data)
    elif family == 'butter':
        b, a = butter(N=N, Wn=Wn, btype=btype, analog=False)
        data = filtfilt(b, a, data)
    else:
        print("[ERROR]: Unknown filter family: %s" % (family))
        sys.exit(-1)

    return data
# end of filter_timeseries

def interp(data, samples, old_dt, new_dt,
           debug=False, debug_plot=None):
    """
    Calls the sinc interp method
    """
    if debug:
        print("[INFO]: Interpolating timeseries: old_dt: %.3f - new_dt: %.3f" %
              (old_dt, new_dt))

    if old_dt == new_dt:
        # Nothing to do!
        return data

    old_times = np.arange(0, samples * old_dt, old_dt)
    if old_times.size == samples + 1:
        old_times = old_times[:-1]

    new_times = np.arange(0, samples * old_dt, new_dt)

    sinc_matrix = (np.tile(new_times, (len(old_times), 1)) -
                   np.tile(old_times[:, np.newaxis], (1, len(new_times))))
    new_data = np.dot(data, np.sinc(sinc_matrix / old_dt))

    if debug:
        # Find data to plot, from t=10s until t=10s+50pts
        old_start_idx = int(10.0 // old_dt) + 1
        old_end_idx = old_start_idx + 50
        if len(old_times) < old_end_idx:
            print("[INFO]: Not enough data to create debug plot!")
            return new_data
        new_start_idx = int(10.0 // new_dt) + 1
        new_end_idx = int(old_times[old_end_idx] // new_dt) + 1

        # Initialize plot
        fig, _ = plt.subplots()
        fig.clf()

        plt.plot(old_times[old_start_idx:old_end_idx],
                 data[old_start_idx:old_end_idx], 'o',
                 new_times[new_start_idx:new_end_idx],
                 new_data[new_start_idx:new_end_idx], 'x')
        plt.grid(True)
        plt.xlabel('Seconds')
        plt.title(os.path.splitext(os.path.basename(debug_plot))[0])
        plt.savefig(debug_plot, format='png',
                    transparent=False, dpi=300)
        pylab.close()

    return new_data

def process_station_dt(station, common_dt, fmax,
                       debug=False, debug_plots_base=None):
    """
    Process the station to set a common dt
    """
    for i in range(0, 3):
        if type(station[i].orientation) is not str:
            debug_orientation = "%03d" % (int(station[i].orientation))
        else:
            debug_orientation = station[i].orientation
        station[i] = process_timeseries_dt(station[i], common_dt,
                                           fmax, debug=debug,
                                           debug_plots_base="%s.%s" %
                                           (debug_plots_base,
                                            debug_orientation))
    return station
#end process_station_dt

def process_timeseries_dt(timeseries, new_dt, fmax,
                          debug=False, debug_plots_base=None):
    """
    Processes a timeseries:
    First filter the data using a lowpass filter using fmax,
    Then adjust the dt to the specified new_dt.
    """
    # call low_pass filter at fmax
    timeseries = filter_timeseries(timeseries, family='butter',
                                   btype='lowpass', fmax=fmax,
                                   N=4, rp=0.1, rs=100, debug=debug)

    # interpolate
    timeseries.acc = interp(timeseries.acc,
                            timeseries.samples,
                            timeseries.dt,
                            new_dt, debug=debug,
                            debug_plot="%s.acc.png" % (debug_plots_base))
    timeseries.vel = interp(timeseries.vel,
                            timeseries.samples,
                            timeseries.dt,
                            new_dt, debug=debug,
                            debug_plot="%s.vel.png" % (debug_plots_base))
    timeseries.dis = interp(timeseries.dis,
                            timeseries.samples,
                            timeseries.dt,
                            new_dt, debug=debug,
                            debug_plot="%s.dis.png" % (debug_plots_base))

    timeseries.samples = timeseries.acc.size
    timeseries.dt = new_dt

    return timeseries
# end of process_timeseries_dt

def check_station_data(station):
    """
    Checks the station's data for empty arrays or NaNs.
    Useful for identifying issues after processing.

    Returns False if any problems found
    """
    for i in range(0, len(station)):
        timeseries = station[i]

        if timeseries.acc.size == 0:
            print("[ERROR]: Empty array after processing timeseries.")
            return False
        if timeseries.vel.size == 0:
            print("[ERROR]: Empty array after processing timeseries.")
            return False
        if timeseries.dis.size == 0:
            print("[ERROR]: Empty array after processing timeseries.")
            return False
        if np.isnan(np.sum(timeseries.acc)):
            print("[ERROR]: NaN data after processing timeseries.")
            return False
        if np.isnan(np.sum(timeseries.vel)):
            print("[ERROR]: NaN data after processing timeseries.")
            return False
        if np.isnan(np.sum(timeseries.dis)):
            print("[ERROR]: NaN data after processing timeseries.")
            return False
    return station
# end of check_data
