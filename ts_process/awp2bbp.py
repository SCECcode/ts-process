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

Utility to convert AWP time history files to BBP format
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse
import numpy as np
from ts_library import integrate, derivative, TimeseriesComponent, rotate_timeseries

def get_dt(input_file):
    """
    Read AWP file and return DT
    """
    val1 = None
    val2 = None
    file_dt = None

    # Figure out dt first, we need it later
    ifile = open(input_file)
    for line in ifile:
        # Skip comments
        if line.startswith("#") or line.startswith("%"):
            continue
        pieces = line.split()
        pieces = [float(piece) for piece in pieces]
        if val1 is None:
            val1 = pieces[0]
            continue
        if val2 is None:
            val2 = pieces[0]
            break
    ifile.close()

    # Quit if cannot figure out dt
    if val1 is None or val2 is None:
        print("[ERROR]: Cannot determine dt from AWP file! Exiting...")
        sys.exit(1)

    # Calculate dt
    file_dt = val2 - val1

    return file_dt
# end get_dt

def read_awp(input_file):
    """
    Reads the input file in awp format and returns arrays containing
    vel_ns, vel_ew, vel_ud components
    """
    time = []
    vel_ns = []
    vel_ew = []
    vel_ud = []

    # Get AWP file dt
    delta_t = get_dt(input_file)
    current_time = 0.0

    try:
        input_fp = open(input_file, 'r')
        for line in input_fp:
            line = line.strip()
            if line.startswith("#") or line.startswith("%"):
                continue
            pieces = line.split()
            pieces = [float(piece) for piece in pieces]
            # Add values to out arrays
            # Note that in AWP files, channels are EW/NS/UD instead of NS/EW/UD
            time.append(current_time)
            current_time = current_time + delta_t
            vel_ew.append(pieces[1])
            vel_ns.append(pieces[2])
            vel_ud.append(pieces[3])
    except IOError as e:
        print(e)
        sys.exit(-1)

    # All done
    input_fp.close()

    # Convert to NumPy Arrays
    time = np.array(time)
    vel_ew = np.array(vel_ew)
    vel_ns = np.array(vel_ns)
    vel_ud = np.array(vel_ud)

    return delta_t, time, vel_ns, vel_ew, vel_ud

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
    out_fp.write("# orientation= %s\n" % (args.orientation))
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

def awp2bbp_main():
    """
    Script to convert AWP files to BBP format
    """
    parser = argparse.ArgumentParser(description="Converts an AWP "
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
    parser.add_argument("input_file", help="AWP input timeseries")
    parser.add_argument("output_stem",
                        help="output BBP filename stem without the "
                        " .{dis,vel,acc}.bbp extensions")
    parser.add_argument("-d", dest="output_dir", default="",
                        help="output directory for the BBP file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file_dis = "%s.dis.bbp" % (os.path.join(args.output_dir,
                                                   args.output_stem))
    output_file_vel = "%s.vel.bbp" % (os.path.join(args.output_dir,
                                                   args.output_stem))
    output_file_acc = "%s.acc.bbp" % (os.path.join(args.output_dir,
                                                   args.output_stem))

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

    # Read AWP file
    print("[INFO]: Reading file %s ..." % (os.path.basename(input_file)))
    delta_t, times, vel_h1, vel_h2, vel_ver = read_awp(input_file)

    # Calculate displacement
    dis_h1 = integrate(vel_h1, delta_t)
    dis_h2 = integrate(vel_h2, delta_t)
    dis_ver = integrate(vel_ver, delta_t)

    # Calculate acceleration
    acc_h1 = derivative(vel_h1, delta_t)
    acc_h2 = derivative(vel_h2, delta_t)
    acc_ver = derivative(vel_ver, delta_t)

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

    # Write header
    o_dis_file = open(output_file_dis, 'w')
    o_vel_file = open(output_file_vel, 'w')
    o_acc_file = open(output_file_acc, 'w')
    write_bbp_header(o_dis_file, "displacement", 'm', args)
    write_bbp_header(o_vel_file, "velocity", 'm/s', args)
    write_bbp_header(o_acc_file, "acceleration", 'm/s^2', args)

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
    awp2bbp_main()
# end of main program
