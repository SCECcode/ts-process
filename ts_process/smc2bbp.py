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

Utility to convert SMC observation files to BBP format
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse
import numpy as np

# Import seismtools needed classes
from ts_library import TimeseriesComponent, baseline_function, \
    rotate_timeseries, check_station_data, integrate, G2CMSS

def parse_arguments():
    """
    This function takes care of parsing the command-line arguments and
    asking the user for any missing parameters that we need
    """
    parser = argparse.ArgumentParser(description="Converts V1/V2 "
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

def read_data(signal):
    """
    The function is to convert signal data into an numpy array of float numbers
    """
    # avoid negative number being stuck
    signal = signal.replace('-', ' -')
    signal = signal.split()

    data = []
    for s in signal:
        data.append(float(s))
    data = np.array(data)
    return data

def read_smc_v1(input_file):
    """
    Reads and processes a V1 file
    """
    record_list = []

    # Loads station into a string
    try:
        fp = open(input_file, 'r')
    except IOError as e:
        print("[ERROR]: opening input file %s" % (input_file))
        return False

    # Print status message
    print("[READING]: %s..." % (input_file))

    # Read data
    channels = fp.read()
    fp.close()

    # Splits the string by channels
    channels = channels.split('/&')
    del(channels[len(channels)-1])

    # Splits the channels
    for i in range(len(channels)):
        channels[i] = channels[i].split('\n')

    # Clean the first row in all but the first channel
    for i in range(1, len(channels)):
        del channels[i][0]

    for i in range(len(channels)):
        # Check this is the uncorrected acceleration data
        ctype = channels[i][0][0:24].lower()
        if ctype != "uncorrected accelerogram":
            print("[ERROR]: processing uncorrected accelerogram ONLY.")
            return False
        else:
            dtype = 'a'

        network = input_file.split('/')[-1].split('.')[0][0:2].upper()
        station_id = input_file.split('/')[-1].split('.')[0][2:].upper()

        # Get location's latitude and longitude
        tmp = channels[i][4].split()
        latitude = tmp[3][:-1]
        longitude = tmp[4]

        # Get station name
        station_name = channels[i][5][0:40].strip()

        # Get orientation, convert to int if it's digit
        tmp = channels[i][6].split()
        orientation = tmp[2]
        if orientation.isdigit():
            orientation = float(int(orientation))
            if orientation == 360:
                orientation = 0.0
        else:
            orientation = orientation.lower()

        # Get date and time; set to fixed format
        start_time = channels[i][3][37:80].split()
        date = start_time[2][:-1]

        tmp = channels[i][14].split()
        hour = tmp[0]
        minute = tmp[1]
        seconds = tmp[2]
        fraction = tmp[3]
        tzone = channels[i][3].split()[-2]
        time = "%s:%s:%s.%s %s" % (hour, minute, seconds, fraction, tzone)

        # Get number of samples and dt
        tmp = channels[i][27].split()
        samples = int(tmp[0])
        delta_t = 1.0 / int(tmp[4])

        # Get signals' data
        tmp = channels[i][28:]
        signal = str()
        for s in tmp:
            signal += s
        acc_data_g = read_data(signal)
        # Convert from g to cm/s/s
        acc_data = acc_data_g * G2CMSS
        # Now integrate to get velocity and displacement
        vel_data = integrate(acc_data, delta_t)
        dis_data = integrate(vel_data, delta_t)

        print("[PROCESSING]: Found component: %s" % (orientation))
        record_list.append(TimeseriesComponent(samples, delta_t, orientation,
                                               acc_data, vel_data, dis_data))

    station_metadata = {}
    station_metadata['network'] = network
    station_metadata['station_id'] = station_id
    station_metadata['type'] = "V1"
    station_metadata['date'] = date
    station_metadata['time'] = time
    station_metadata['longitude'] = longitude
    station_metadata['latitude'] = latitude
    station_metadata['high_pass'] = -1
    station_metadata['low_pass'] = -1

    return record_list, station_metadata

def read_smc_v2(input_file):
    """
    Reads and processes a V2 file
    """
    record_list = []

    # Loads station into a string
    try:
        fp = open(input_file, 'r')
    except IOError as e:
        print("[ERROR]: opening input file %s" % (input_file))
        return False

    # Print status message
    print("[READING]: %s..." % (input_file))

    # Read data
    channels = fp.read()
    fp.close()

    # Splits the string by channels
    channels = channels.split('/&')
    del(channels[len(channels)-1])

    # Splits the channels
    for i in range(len(channels)):
        channels[i] = channels[i].split('\n')

    # Clean the first row in all but the first channel
    for i in range(1, len(channels)):
        del channels[i][0]

    for i in range(len(channels)):
        tmp = channels[i][0].split()
        # Check this is the corrected acceleration data
        ctype = (tmp[0] + " " + tmp[1]).lower()
        if ctype != "corrected accelerogram":
            print("[ERROR]: processing corrected accelerogram ONLY.")
            return False

        # Get network code and station id
        network = input_file.split('/')[-1].split('.')[0][0:2].upper()
        station_id = input_file.split('/')[-1].split('.')[0][2:].upper()

        # Get location's latitude and longitude
        tmp = channels[i][5].split()
        latitude = tmp[3][:-1]
        longitude = tmp[4]

        # Make sure we captured the right values
        if latitude[-1].upper() != "N" and latitude.upper() != "S":
            # Maybe it is an old file, let's try to get the values again...
            latitude = (float(tmp[3]) +
                        (float(tmp[4]) / 60.0) +
                        (float(tmp[5][:-2]) / 3600.0))
            latitude = "%s%s" % (str(latitude), tmp[5][-2])
            longitude = (float(tmp[6]) +
                         (float(tmp[7]) / 60.0) +
                         (float(tmp[8][:-1]) / 3600.0))
            longitude = "%s%s" % (str(longitude), tmp[8][-1])

        # Get orientation from integer header
        orientation = float(int(channels[i][26][50:55]))
        if orientation == 360:
            orientation = 0.0
        elif orientation == 500:
            orientation = "up"
        elif orientation == 600:
            orientation = "down"

        # Get filtering information
        tmp = channels[i][14].split()
        high_pass = float(tmp[8])
        low_pass = float(tmp[10])

        # Get station name
        station_name = channels[i][6][0:40].strip()

        # Get date and time; set to fixed format
        start_time = channels[i][4][37:80].split()
        try:
            date = start_time[2][:-1]
            tmp = start_time[3].split(':')
            hour = tmp[0]
            minute = tmp[1]
            seconds, fraction = tmp[2].split('.')
            # Works for both newer and older V2 files
            tzone = channels[i][4].split()[5]
        except IndexError:
            date = '00/00/00'
            hour = '00'
            minute = '00'
            seconds = '00'
            fraction = '0'
            tzone = '---'

        # Put it all together
        time = "%s:%s:%s.%s %s" % (hour, minute, seconds, fraction, tzone)

        # Get number of samples and dt
        tmp = channels[i][45].split()
        samples = int(tmp[0])
        delta_t = float(tmp[8])

        # Get signals' data
        tmp = channels[i][45:]
        a_signal = str()
        v_signal = str()
        d_signal = str()

        for s in tmp:
            # Detecting separate line and get data type
            if "points" in s.lower():
                line = s.split()
                if line[3].lower() == "accel" or line[3].lower() == "acc":
                    dtype = 'a'
                elif line[3].lower() == "veloc" or line[3].lower() == "vel":
                    dtype = 'v'
                elif line[3].lower() == "displ" or line[3].lower() == "dis":
                    dtype = 'd'
                else:
                    dtype = "unknown"

            # Processing data
            else:
                if dtype == 'a':
                    a_signal += s
                elif dtype == 'v':
                    v_signal += s
                elif dtype == 'd':
                    d_signal += s

        acc_data = read_data(a_signal)
        vel_data = read_data(v_signal)
        dis_data = read_data(d_signal)

        print("[PROCESSING]: Found component: %s" % (orientation))
        record_list.append(TimeseriesComponent(samples, delta_t, orientation,
                                               acc_data, vel_data, dis_data))

    station_metadata = {}
    station_metadata['network'] = network
    station_metadata['station_id'] = station_id
    station_metadata['type'] = "V2"
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
        # We haven't added any padding to the timeseries yet
        out_fp.write("#     padding= 0\n")
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

def smc2bbp_process(input_file, output_dir):
    """
    Converts input_file to bbp format
    """
    if (input_file.upper().endswith(".RAW") or
        input_file.upper().endswith(".V1")):
        station, station_metadata = read_smc_v1(input_file)
    else:
        # Must be a ".V2" file!
        station, station_metadata = read_smc_v2(input_file)

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

def smc2bbp_main():
    """
    Main function for the smc2bbp conversion utility
    """
    args = parse_arguments()

    if args.infile is not None:
        # Only one file to process
        process_list = [args.infile]
    else:
        # Create list of files to process
        process_list = []
        for item in os.listdir(args.indir):
            if (item.upper().endswith(".V1") or
                item.upper().endswith(".RAW") or
                item.upper().endswith(".V2")):
                process_list.append(os.path.join(args.indir,
                                                 item))

    # Now process the list of files
    for item in process_list:
        smc2bbp_process(item, args.outdir)

# ============================ MAIN ==============================
if __name__ == "__main__":
    smc2bbp_main()
# end of main program
