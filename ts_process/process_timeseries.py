#!/usr/bin/env python3
"""
BSD 3-Clause License

Copyright (c) 2020, Southern California Earthquake Center
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

The program is to read input seismograms; process their signals.
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse

from file_utilities import write_bbp, read_stamp, read_files
from ts_library import process_station_dt, check_station_data, seism_cutting, seism_appendzeros

def synchronize_all_stations(obs_data, stations, stamp, eqtimestamp, leading):
    """
    synchronize the stating time and ending time of data arrays
    obs_data = recorded data (optional); stations = simulation signal(s)
    """
    # If we have a recorded data time stamp
    if stamp is not None and obs_data is not None:
        start = stamp[0]*3600 + stamp[1]*60 + stamp[2]
        eq_time = eqtimestamp[0]*3600 + eqtimestamp[1]*60 + eqtimestamp[2]
        sim_start = eq_time - leading

        for i in range(0, 3):
            # synchronize the start time
            if start < sim_start:
                # data time < sim time < earthquake time; cutting data array
                obs_data[i] = seism_cutting('front', (sim_start - start),
                                            20, obs_data[i])
            elif start > eq_time:
                # sim time < earthquake time < data time; adding zeros in front
                obs_data[i] = seism_appendzeros('front', (start - eq_time),
                                                20, obs_data[i])
                for station in stations:
                    station[i] = seism_cutting('front', (eq_time - sim_start),
                                               20, station[i])
            else:
                # sim time < data time < earthquake time; adding zeros
                obs_data[i] = seism_appendzeros('front', (start - sim_start),
                                                20, obs_data[i])

    # synchronize the ending time
    if obs_data is not None:
        obs_dt = obs_data[0].dt
        obs_samples = obs_data[0].samples
        obs_time = obs_dt * obs_samples
    else:
        obs_time = None

    # Find target timeseries duration
    target_time = None
    if obs_time is not None:
        target_time = obs_time
    for station in stations:
        station_dt = station[0].dt
        station_samples = station[0].samples
        station_time = station_dt * station_samples
        if target_time is None:
            target_time = station_time
            continue
        target_time = min(target_time, station_time)

    # Work on obs_data
    if obs_data is not None:
        for i in range(0, 3):
            if obs_time > target_time:
                obs_data[i] = seism_cutting('end', (obs_time - target_time),
                                            20, obs_data[i])
        obs_samples = obs_data[0].samples
        obs_time = obs_dt * obs_samples

    # Work on simulated data
    for station in stations:
        for i in range(0, 3):
            sim_dt = station[i].dt
            sim_samples = station[i].samples
            sim_time = sim_dt * sim_samples
            if sim_time > target_time:
                station[i] = seism_cutting('end', (sim_time - target_time),
                                           20, station[i])

    # scale the data if they have one sample in difference after synchronizing
    total_samples = None
    if obs_data is not None:
        total_samples = obs_samples
    for station in stations:
        sim_samples = station[0].samples
        if total_samples is None:
            total_samples = sim_samples
            continue
        total_samples = max(sim_samples, total_samples)

    # For obs_data
    if obs_data is not None:
        for i in range(0, 3):
            if obs_data[i].samples == total_samples - 1:
                obs_data[i] = seism_appendzeros('end', obs_data[i].dt,
                                                20, obs_data[i])
    # For simulated data
    for station in stations:
        for i in range(0, 3):
            if station[i].samples == total_samples - 1:
                station[i] = seism_appendzeros('end', station[i].dt,
                                               20, station[i])

    return obs_data, stations
# end of synchronize_all_stations

def process(obs_file, obs_data, input_files, stations, params):
    """
    This method processes the signals in each pair of stations.
    Processing consists on scaling, low-pass filtering, alignment
    and other things to make both signals compatible to apply GOF method.
    obs_data: recorded data
    stations: simulation
    """
    # Process signals to have the same dt
    if obs_data is not None:
        debug_plots_base = os.path.join(params['outdir'],
                                        os.path.basename(obs_file).split('.')[0])
        obs_data = process_station_dt(obs_data,
                                      params['targetdt'],
                                      params['lp'],
                                      taper=params['taper'],
                                      debug=params['debug'],
                                      debug_plots_base=debug_plots_base)
    new_stations = []
    for station, input_file in zip(stations, input_files):
        debug_plots_base = os.path.join(params['outdir'],
                                        os.path.basename(input_file).split('.')[0])
        new_station = process_station_dt(station,
                                         params['targetdt'],
                                         params['lp'],
                                         taper=params['taper'],
                                         debug=params['debug'],
                                         debug_plots_base=debug_plots_base)
        new_stations.append(new_station)
    stations = new_stations

    # Read obs_file timestamp if needed
    stamp = None
    if obs_data is not None:
        stamp = read_stamp(obs_file)

    # Synchronize starting and ending time of data arrays
    obs_data, stations = synchronize_all_stations(obs_data,
                                                  stations,
                                                  stamp,
                                                  params['eq_time'],
                                                  params['leading'])

    # Check number of samples
    if obs_data is not None:
        num_samples = obs_data[0].samples
    else:
        num_samples = stations[0][0].samples

    for station in stations:
        if station[0].samples != num_samples:
            print("[ERROR]: two timseries do not have the same number"
                  " of samples after processing.")
            sys.exit(-1)

    # Check the data
    if obs_data is not None:
        if not check_station_data(obs_data):
            print("[ERROR]: processed recorded data contains errors!")
            sys.exit(-1)
    for station in stations:
        if not check_station_data(station):
            print("[ERROR]: processed simulated data contains errors!")
            sys.exit(-1)

    # All done
    return obs_data, stations
# end of process

def parse_arguments():
    """
    This function takes care of parsing the command-line arguments and
    asking the user for any missing parameters that we need
    """
    parser = argparse.ArgumentParser(description="Processes a number of "
                                     "timeseries files and prepares them "
                                     "for plotting.")
    parser.add_argument("--obs", dest="obs_file",
                        help="input file containing recorded data")
    parser.add_argument("--leading", type=float, dest="leading",
                        help="leading time for the simulation (seconds)")
    parser.add_argument("--eq-time", dest="eq_time",
                        help="earthquake start time (HH:MM:SS.CCC)")
    parser.add_argument("--dt", type=float, dest="targetdt",
                        help="target dt for all processed signals")
    parser.add_argument("--lp-freq", type=float, dest="lp",
                        help="frequency for low-pass filter")
    parser.add_argument("--taper", type=int, dest="taper",
                        help="taper window length, default is 8")
    parser.add_argument("--output-dir", dest="outdir",
                        help="output directory for the outputs")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="produces debug plots and outputs steps in detail")
    parser.add_argument('input_files', nargs='*')
    args = parser.parse_args()

    # Input files
    files = args.input_files
    obs_file = args.obs_file

    if len(files) < 1 or len(files) == 1 and obs_file is None:
        print("[ERROR]: Please provide at least two timeseries to process!")
        sys.exit(-1)

    # Check for missing input parameters
    params = {}

    if args.outdir is None:
        print("[ERROR]: Please provide output directory!")
    else:
        params['outdir'] = args.outdir

    # Check for user-provided taper window length
    if args.taper is None:
        params['taper'] = 8
    else:
        params['taper'] = args.taper

    # None means no low-pass filtering after adjusting dt
    params['lp'] = args.lp

    if args.targetdt is None:
        print("[ERROR]: Please provide a target DT to be used in all signals!")
    else:
        params['targetdt'] = args.targetdt

    if args.eq_time is None:
        print("[ERROR]: Please provide earthquake time!")
    else:
        tokens = args.eq_time.split(':')
        if len(tokens) < 3:
            print("[ERROR]: Invalid time format!")
            sys.exit(-1)
        try:
            params['eq_time'] = [float(token) for token in tokens]
        except ValueError:
            print("[ERROR]: Invalid time format!")
            sys.exit(-1)

    if args.leading is None:
        print("[ERROR]: Please enter the simulation leading time!")
    else:
        params['leading'] = args.leading

    params['debug'] = args.debug is not None

    return obs_file, files, params

def process_main():
    """
    Main function for processing seismograms
    """
    # First let's get all aruments that we need
    obs_file, input_files, params = parse_arguments()

    # Read input files
    obs_data, stations = read_files(obs_file, input_files)

    # Process signals
    obs_data, stations = process(obs_file, obs_data,
                                 input_files, stations,
                                 params)

    # Write processed files
    if obs_data is not None:
        obs_file_out = os.path.join(params['outdir'],
                                    "p-%s" % os.path.basename(obs_file))
        write_bbp(obs_file, obs_file_out, obs_data, params)

    for input_file, station in zip(input_files, stations):
        out_file = os.path.join(params['outdir'],
                                "p-%s" % os.path.basename(input_file))
        write_bbp(input_file, out_file, station, params)
# end of process_main

# ============================ MAIN ==============================
if __name__ == "__main__":
    process_main()
# end of main program
