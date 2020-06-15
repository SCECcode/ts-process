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

Utility to convert Hercules .her time history files to BBP format
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse
import numpy as np
from ts_library import TimeseriesComponent, rotate_timeseries

def parse_her_header(filename):
    """
    This function parses the her file header
    to try to figure out what units to use
    """
    # Default unit is meters
    unit = "m"

    try:
        input_file = open(filename, 'r')
        for line in input_file:
            line = line.strip()
            if line.startswith("#"):
                # Header line, look into it
                pieces = line.split()
                if len(pieces) != 11:
                    # Not the line we are looking for
                    continue
                if pieces[2].find("(m)") > 0:
                    # It's meters!
                    unit = "m"
                    break
                if pieces[2].find("(cm)") > 0:
                    # It's cm!
                    unit = "cm"
                    break
            continue
    except IOError:
        print("[ERROR]: Unable to read file: %s" % (filename))
        sys.exit(1)
    input_file.close()

    # Return units
    return unit

def write_bbp_header(out_fp, file_type, file_unit, args):
    """
    This function writes the bbp header
    """
    orientation = args.orientation.strip()
    orientations = orientation.split(",")
    orientations = [val.strip() for val in orientations]

    # Write header
    out_fp.write("#     Station= %s\n" % (args.station_name))
    out_fp.write("#        time= %s\n" % (args.time))
    out_fp.write("#         lon= %s\n" % (args.longitude))
    out_fp.write("#         lat= %s\n" % (args.latitude))
    out_fp.write("#       units= %s\n" % (file_unit))
    out_fp.write("#     padding= 0\n")
    out_fp.write("# orientation= %s\n" % (orientation))
    out_fp.write("#\n")
    out_fp.write("# Data fields are TAB-separated\n")
    out_fp.write("# Column 1: Time (s)\n")
    out_fp.write("# Column 2: H1 component ground "
                 "%s (+ is %s)\n" % (file_type, orientations[0]))
    out_fp.write("# Column 3: H2 component ground "
                 "%s (+ is %s)\n" % (file_type, orientations[1]))
    out_fp.write("# Column 4: V component ground "
                 "%s (+ is %s)\n" % (file_type, orientations[2]))
    out_fp.write("#\n")

def read_hercules(input_file):
    """
    Reads the input hercules file and returns the
    data along with parsed header lines
    """
    times = []
    acc_h1 = []
    vel_h1 = []
    dis_h1 = []
    acc_h2 = []
    vel_h2 = []
    dis_h2 = []
    acc_ver = []
    vel_ver = []
    dis_ver = []
    dis_header = []
    vel_header = []
    acc_header = []

    try:
        input_fp = open(input_file, 'r')
        for line in input_fp:
            line = line.strip()
            # Skip comments
            if line.startswith("#") or line.startswith("%"):
                pieces = line.split()[1:]
                # Write header
                if len(pieces) >= 10:
                    dis_header.append("# her header: # %s %s %s %s\n" %
                                     (pieces[0], pieces[1], pieces[2], pieces[3]))
                    vel_header.append("# her header: # %s %s %s %s\n" %
                                     (pieces[0], pieces[4], pieces[5], pieces[6]))
                    acc_header.append("# her header: # %s %s %s %s\n" %
                                     (pieces[0], pieces[7], pieces[8], pieces[9]))
                else:
                    dis_header.append("# her header: %s\n" % (line))
                continue
            pieces = line.split()
            pieces = [float(piece) for piece in pieces]
            # Write timeseries to files. Please not that Hercules files have
            # the vertical component positive pointing down so we have to flip it
            # here to match the BBP format in which vertical component points up
            times.append(pieces[0])
            dis_h1.append(pieces[1])
            dis_h2.append(pieces[2])
            dis_ver.append(-1 * pieces[3])
            vel_h1.append(pieces[4])
            vel_h2.append(pieces[5])
            vel_ver.append(-1 * pieces[6])
            acc_h1.append(pieces[7])
            acc_h2.append(pieces[8])
            acc_ver.append(-1 * pieces[9])
    except IOError as e:
        print(e)
        sys.exit(-1)

    # All done
    input_fp.close()

    # Convert to NumPy Arrays
    times = np.array(times)
    vel_h1 = np.array(vel_h1)
    vel_h2 = np.array(vel_h2)
    vel_ver = np.array(vel_ver)
    acc_h1 = np.array(acc_h1)
    acc_h2 = np.array(acc_h2)
    acc_ver = np.array(acc_ver)
    dis_h1 = np.array(dis_h1)
    dis_h2 = np.array(dis_h2)
    dis_ver = np.array(dis_ver)

    delta_t = times[1] - times[0]

    # Group headers
    headers = [dis_header, vel_header, acc_header]

    return (headers, delta_t, times,
            acc_h1, acc_h2, acc_ver,
            vel_h1, vel_h2, vel_ver,
            dis_h1, dis_h2, dis_ver)

def her2bbp_main():
    """
    Main function for her to bbp converter
    """
    parser = argparse.ArgumentParser(description="Converts a Hercules .her"
                                     "file to BBP format, generating "
                                     "displacement, velocity and acceleration "
                                     "BBP files.")
    parser.add_argument("-s", "--station-name", dest="station_name",
                        default="NoName",
                        help="provides the name for this station")
    parser.add_argument("--lat", dest="latitude", type=float, default=0.0,
                        help="provides the latitude for the station")
    parser.add_argument("--lon", dest="longitude", type=float, default=0.0,
                        help="provides the longitude for the station")
    parser.add_argument("-t", "--time", default="00/00/00,0:0:0.0 UTC",
                        help="provides timing information for this timeseries")
    parser.add_argument("-o", "--orientation", default="0,90,UP",
                        dest="orientation",
                        help="orientation, default: 0,90,UP")
    parser.add_argument("--azimuth", type=float, dest="azimuth",
                        help="azimuth for rotation (degrees)")
    parser.add_argument("input_file", help="Hercules input timeseries")
    parser.add_argument("output_stem",
                        help="output BBP filename stem without the "
                        " .{dis,vel,acc}.bbp extensions")
    parser.add_argument("-d", dest="output_dir", default="",
                        help="output directory for the BBP file")
    args = parser.parse_args()

    # Check orientation
    orientation = args.orientation.split(",")
    if len(orientation) != 3:
        print("[ERROR]: Need to specify orientation for all 3 components!")
        sys.exit(-1)
    orientation[0] = float(orientation[0])
    orientation[1] = float(orientation[1])
    orientation[2] = orientation[2].lower()
    if orientation[2] != "up" and orientation[2] != "down":
        print("[ERROR]: Vertical orientation must be up or down!")
        sys.exit(-1)

    input_file = args.input_file
    output_file_dis = "%s.dis.bbp" % (os.path.join(args.output_dir,
                                                   args.output_stem))
    output_file_vel = "%s.vel.bbp" % (os.path.join(args.output_dir,
                                                   args.output_stem))
    output_file_acc = "%s.acc.bbp" % (os.path.join(args.output_dir,
                                                   args.output_stem))

    # Try to get the units used in the her file
    units = {"m": ["m", "m/s", "m/s^2"],
             "cm": ["cm", "cm/s", "cm/s^2"]}
    unit = parse_her_header(input_file)

    # Covert from her to BBP format
    print("[INFO]: Reading file %s ..." % (os.path.basename(input_file)))

    (headers, delta_t, times,
     acc_h1, acc_h2, acc_ver,
     vel_h1, vel_h2, vel_ver,
     dis_h1, dis_h2, dis_ver) = read_hercules(input_file)

    # Create station data structures
    samples = vel_h1.size

    # samples, dt, data, acceleration, velocity, displacement
    signal_h1 = TimeseriesComponent(samples, delta_t, orientation[0],
                                    acc_h1, vel_h1, dis_h1)
    signal_h2 = TimeseriesComponent(samples, delta_t, orientation[1],
                                    acc_h2, vel_h2, dis_h2)
    signal_ver = TimeseriesComponent(samples, delta_t, orientation[2],
                                     acc_ver, vel_ver, dis_ver)
    station = [signal_h1, signal_h2, signal_ver]

    # Rotate timeseries if needed
    if args.azimuth is not None:
        print("[INFO]: Rotating timeseries - %f degrees" % (args.azimuth))
        station = rotate_timeseries(station, args.azimuth)

    # Update orientation after rotation so headers reflect any changes
    args.orientation = "%s,%s,%s" % (str(station[0].orientation),
                                     str(station[1].orientation),
                                     str(station[2].orientation))

    # Pull data back
    acc_h1 = station[0].acc.tolist()
    vel_h1 = station[0].vel.tolist()
    dis_h1 = station[0].dis.tolist()
    acc_h2 = station[1].acc.tolist()
    vel_h2 = station[1].vel.tolist()
    dis_h2 = station[1].dis.tolist()
    acc_ver = station[2].acc.tolist()
    vel_ver = station[2].vel.tolist()
    dis_ver = station[2].dis.tolist()

    o_dis_file = open(output_file_dis, 'w')
    o_vel_file = open(output_file_vel, 'w')
    o_acc_file = open(output_file_acc, 'w')
    write_bbp_header(o_dis_file, "displacement", units[unit][0], args)
    write_bbp_header(o_vel_file, "velocity", units[unit][1], args)
    write_bbp_header(o_acc_file, "acceleration", units[unit][2], args)

    # Write headers from original Hercules file
    dis_header = headers[0]
    vel_header = headers[1]
    acc_header = headers[2]

    for line in dis_header:
        o_dis_file.write(line)
    for line in vel_header:
        o_vel_file.write(line)
    for line in acc_header:
        o_acc_file.write(line)

    # Write files
    for (time, disp_h1, disp_h2, disp_ver,
         velo_h1, velo_h2, velo_ver,
         accel_h1, accel_h2, accel_ver) in zip(times, dis_h1, dis_h2, dis_ver,
                                               vel_h1, vel_h2, vel_ver,
                                               acc_h1, acc_h2, acc_ver):
        o_dis_file.write("%1.9E %1.9E %1.9E %1.9E\n" %
                         (time, disp_h1, disp_h2, disp_ver))
        o_vel_file.write("%1.9E %1.9E %1.9E %1.9E\n" %
                         (time, velo_h1, velo_h2, velo_ver))
        o_acc_file.write("%1.9E %1.9E %1.9E %1.9E\n" %
                         (time, accel_h1, accel_h2, accel_ver))

    # All done
    o_dis_file.close()
    o_vel_file.close()
    o_acc_file.close()

# ============================ MAIN ==============================
if __name__ == "__main__":
    her2bbp_main()
# end of main program
