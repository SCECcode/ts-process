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

Utility to convert Hercules .her time history files to BBP format
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse

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
    parser.add_argument("input_file", help="Hercules input timeseries")
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

    # Try to get the units used in the her file
    units = {"m": ["m", "m/s", "m/s^2"],
             "cm": ["cm", "cm/s", "cm/s^2"]}
    unit = parse_her_header(input_file)

    # Covert from her to BBP format
    ifile = open(input_file)
    o_dis_file = open(output_file_dis, 'w')
    o_vel_file = open(output_file_vel, 'w')
    o_acc_file = open(output_file_acc, 'w')
    write_bbp_header(o_dis_file, "displacement", units[unit][0], args)
    write_bbp_header(o_vel_file, "velocity", units[unit][1], args)
    write_bbp_header(o_acc_file, "acceleration", units[unit][2], args)
    for line in ifile:
        line = line.strip()
        # Skip comments
        if line.startswith("#") or line.startswith("%"):
            pieces = line.split()[1:]
            # Write header
            if len(pieces) >= 10:
                o_dis_file.write("# her header: # %s %s %s %s\n" %
                                 (pieces[0], pieces[1], pieces[2], pieces[3]))
                o_vel_file.write("# her header: # %s %s %s %s\n" %
                                 (pieces[0], pieces[4], pieces[5], pieces[6]))
                o_acc_file.write("# her header: # %s %s %s %s\n" %
                                 (pieces[0], pieces[7], pieces[8], pieces[9]))
            else:
                o_dis_file.write("# her header: %s\n" % (line))
            continue
        pieces = line.split()
        pieces = [float(piece) for piece in pieces]
        # Write timeseries to files. Please not that Hercules files have
        # the vertical component positive pointing down so we have to flip it
        # here to match the BBP format in which vertical component points up
        o_dis_file.write("%1.9E %1.9E %1.9E %1.9E\n" %
                         (pieces[0], pieces[1], pieces[2], -1 * pieces[3]))
        o_vel_file.write("%1.9E %1.9E %1.9E %1.9E\n" %
                         (pieces[0], pieces[4], pieces[5], -1 * pieces[6]))
        o_acc_file.write("%1.9E %1.9E %1.9E %1.9E\n" %
                         (pieces[0], pieces[7], pieces[8], -1 * pieces[9]))

    # All done, close everything
    ifile.close()
    o_dis_file.close()
    o_vel_file.close()
    o_acc_file.close()

# ============================ MAIN ==============================
if __name__ == "__main__":
    her2bbp_main()
# end of main program
