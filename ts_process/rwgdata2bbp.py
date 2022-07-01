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

Utility to convert RWG observation data files to BBP format
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse
import numpy as np

# Import tsprocess needed functions
from file_utilities import read_file_bbp2
from ts_library import TimeseriesComponent, baseline_function, \
    rotate_timeseries, integrate

def parse_arguments():
    """
    This function takes care of parsing the command-line arguments and
    asking the user for any missing parameters that we need
    """
    parser = argparse.ArgumentParser(description="Converts RWG "
                                     " observation files to BBP format.")
    parser.add_argument("-o", "--output", dest="outdir", required=True,
                        help="output directory name")
    parser.add_argument("-i", "--input", dest="infile",
                        help="input file (overrides --dir below)")
    parser.add_argument("-d", "--dir", dest="indir",
                        help="input directory")
    args = parser.parse_args()

    if args.infile is None and args.indir is None:
        print("ERROR: Please specify either an input file or directory!")
        sys.exit(-1)

    if args.infile is not None:
        args.indir = None

    return args

def read_rwg_obs_data(input_file):
    """
    Reads and processes a RWG observation file
    """
    record_list = []

    # Read file
    print("[READING]: %s..." % (input_file))

    # First, read the headers
    headers = []
    try:
        bbp_file = open(input_file, 'r')
        for line in bbp_file:
            line = line.strip()
            if line.startswith('#') or line.startswith('%'):
                headers.append(line)
        bbp_file.close()
    except IOError:
        print("[ERROR]: error reading bbp file: %s" % (input_file))
        sys.exit(1)

    # Now read the data
    [times, vel_h1, vel_h2, vel_ver] = read_file_bbp2(input_file)

    for data, orientation in zip([vel_h1, vel_h2, vel_ver],
                                 [0.0, 90.0, 'up']):

        # Get network code and station id
        basefile = os.path.splitext(os.path.basename(input_file))[0]
        tokens = basefile.split("_")
        network = tokens[0].upper()
        station_id = tokens[1].upper()

        # Get location's latitude and longitude
        latitude = "N/A"
        longitude = "N/A"
        for line in headers:
            if "lon=" in line:
                longitude = float(line.split()[2])
            if "lat=" in line:
                latitude = float(line.split()[2])

        # Get filtering information
        high_pass = 0.1
        low_pass = 5.0

        date = '00/00/00'
        hour = '00'
        minute = '00'
        seconds = '00'
        fraction = '0'
        tzone = '---'

        # Put it all together
        time = "%s:%s:%s.%s %s" % (hour, minute, seconds, fraction, tzone)

        # Get number of samples and dt
        samples = data.size
        delta_t = times[1] - times[0]

        acc_data = data
        vel_data = integrate(acc_data, delta_t)
        dis_data = integrate(vel_data, delta_t)

        print("[PROCESSING]: Found component: %s" % (orientation))
        record_list.append(TimeseriesComponent(samples, delta_t, orientation,
                                               acc_data, vel_data, dis_data))

    station_metadata = {}
    station_metadata['network'] = network
    station_metadata['station_id'] = station_id
    station_metadata['type'] = "RWGOBS"
    station_metadata['date'] = date
    station_metadata['time'] = time
    station_metadata['longitude'] = longitude
    station_metadata['latitude'] = latitude
    station_metadata['high_pass'] = high_pass
    station_metadata['low_pass'] = low_pass

    return record_list, station_metadata

def process_observation_data(station):
    """
    This function processes the observation data
    using baseline correction and rotation (if needed)
    """
    # Validate inputs
    if len(station) != 3:
        print("[ERROR]: Expecting 3 components!")
        return False

    # Reorder components if needed so that vertical is always the last one
    if isinstance(station[0].orientation, str):
        tmp = station[0]
        station[0] = station[2]
        station[2] = tmp
    elif isinstance(station[1].orientation, str):
        tmp = station[1]
        station[1] = station[2]
        station[2] = tmp

    # First we apply the baseline correction, use 5th order polynomial
    order = 5
    # Inputs are in cm/sec2, so no scaling
    gscale = 1.0

    # Apply baseline correction to all components
    for component in station:
        _, new_acc, new_vel, new_dis = baseline_function(component.acc,
                                                         component.dt,
                                                         gscale, order)
        component.acc = new_acc
        component.vel = new_vel
        component.dis = new_dis

    # Now rotate if needed, so that components are 0 and 90 degrees
    # Always pick the smaller angle for rotation
    rotation_angle = min(station[0].orientation,
                         station[1].orientation)
    return rotate_timeseries(station, rotation_angle)

def write_bbp(station, station_metadata, destination):
    """
    This function generates .bbp files for
    each of velocity/acceleration/displacement
    """
    filename_base = ("%s_%s.%s" %
                     (station_metadata['network'],
                      station_metadata['station_id'],
                      station_metadata['type']))

    # round data to 7 decimals in order to print properly
    for component in station:
        if component.orientation in [0, 360, 180, -180]:
            dis_ns = component.dis.tolist()
            vel_ns = component.vel.tolist()
            acc_ns = component.acc.tolist()
        elif component.orientation in [90, -270, -90, 270]:
            dis_ew = component.dis.tolist()
            vel_ew = component.vel.tolist()
            acc_ew = component.acc.tolist()
        elif (component.orientation.upper() == "UP" or
              component.orientation.upper() == "DOWN"):
            dis_up = component.dis.tolist()
            vel_up = component.vel.tolist()
            acc_up = component.acc.tolist()
        else:
            pass

    # Prepare to output
    out_data = [['dis', dis_ns, dis_ew, dis_up, 'displacement', 'cm'],
                ['vel', vel_ns, vel_ew, vel_up, 'velocity', 'cm/s'],
                ['acc', acc_ns, acc_ew, acc_up, 'acceleration', 'cm/s^2']]

    for data in out_data:
        filename = "%s.%s.bbp" % (filename_base, data[0])
        try:
            out_fp = open(os.path.join(destination, filename), 'w')
        except IOError as e:
            print("[ERROR]: Writing BBP file: %s" % (filename))
            return False

        # Start with time = 0.0
        time = [0.000]
        samples = component.samples
        while samples > 1:
            time.append(time[len(time)-1] + component.dt)
            samples -= 1

        # Write header
        out_fp.write("#     Station= %s_%s\n" %
                     (station_metadata['network'],
                      station_metadata['station_id']))
        out_fp.write("#        time= %s,%s\n" %
                     (station_metadata['date'],
                      station_metadata['time']))
        out_fp.write("#         lon= %s\n" %
                     (station_metadata['longitude']))
        out_fp.write("#         lat= %s\n" %
                     (station_metadata['latitude']))
        out_fp.write("#          hp= %s\n" %
                     (station_metadata['high_pass']))
        out_fp.write("#          lp= %s\n" %
                     (station_metadata['low_pass']))
        out_fp.write("#       units= %s\n" % (data[5]))
        # Orientation is always 0,90,UP as we just rotated the timeseries
        out_fp.write("# orientation= 0,90,UP\n")
        out_fp.write("#\n")
        out_fp.write("# Data fields are TAB-separated\n")
        out_fp.write("# Column 1: Time (s)\n")
        out_fp.write("# Column 2: H1 component ground "
                     "%s (+ is 000)\n" % (data[4]))
        out_fp.write("# Column 3: H2 component ground "
                     "%s (+ is 090)\n" % (data[4]))
        out_fp.write("# Column 4: V component ground "
                     "%s (+ is upward)\n" % (data[4]))
        out_fp.write("#\n")

        # Write timeseries
        for val_time, val_ns, val_ew, val_ud in zip(time, data[1],
                                                    data[2], data[3]):
            out_fp.write("%5.7f   %5.9e   %5.9e    %5.9e\n" %
                         (val_time, val_ns, val_ew, val_ud))

        # All done, close file
        out_fp.close()
        print("[WRITING]: Wrote BBP file: %s" % (filename))

def rwgdata2bbp_process(input_file, output_dir):
    """
    Converts input_file to bbp format
    """
    station, station_metadata = read_rwg_obs_data(input_file)

    if station:
        station = process_observation_data(station)
        # Make sure output is valid
        if not station:
            print("[ERROR]: Processing input file: %s" % (input_file))
            return
    else:
        print("[ERROR]: Reading input file: %s" % (input_file))
        return

    # Write BBP file
    write_bbp(station, station_metadata, output_dir)

def rwgdata2bbp_main():
    """
    Main function for the rwgdata2bbp conversion utility
    """
    args = parse_arguments()

    if args.infile is not None:
        # Only one file to process
        process_list = [args.infile]
    else:
        # Create list of files to process
        process_list = []
        for item in os.listdir(args.indir):
            if item.upper().endswith(".BBP"):
                process_list.append(os.path.join(args.indir,
                                                 item))

    # Now process the list of files
    for item in process_list:
        rwgdata2bbp_process(item, args.outdir)

# ============================ MAIN ==============================
if __name__ == "__main__":
    rwgdata2bbp_main()
# end of main program
