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

This program plots a several timeseries together without any processing
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import argparse
import matplotlib as mpl
if mpl.get_backend() != 'agg':
    mpl.use('Agg') # Disables use of Tk/X11

# Import seismtools functions
from file_utilities import read_files
from ts_library import calculate_distance
from ts_plot_library import plot_overlay_timeseries

def parse_arguments():
    """
    This function takes care of parsing the command-line arguments and
    asking the user for any missing parameters that we need
    """
    parser = argparse.ArgumentParser(description="Creates comparison plots of "
                                     " a number of timeseries files.")
    parser.add_argument("-o", "--output", dest="outfile", required=True,
                        help="output png file")
    parser.add_argument("--epicenter-lat", dest="epicenter_lat", type=float,
                        help="earthquake epicenter latitude")
    parser.add_argument("--epicenter-lon", dest="epicenter_lon", type=float,
                        help="earthquake epicenter longitude")
    parser.add_argument("--st-lat", "--station-latitude", dest="st_lat",
                        type=float, help="station latitude")
    parser.add_argument("--st-lon", "--station-longitude", dest="st_lon",
                        type=float, help="station longitude")
    parser.add_argument("-s", "--station-name", "--station", dest="station",
                        help="station name")
    parser.add_argument("--station-list", dest="station_list",
                        help="station list with latitude and longitude")
    parser.add_argument("--xmin", dest="xmin", type=float,
                        help="xmin to plot")
    parser.add_argument("--xmax", dest="xmax", type=float,
                        help="xmax to plot")
    parser.add_argument('input_files', nargs='*')
    args = parser.parse_args()

    if args.st_lat is not None and args.st_lon is not None:
        args.st_loc = [args.st_lat, args.st_lon]
    else:
        args.st_loc = None
    if args.epicenter_lat is not None and args.epicenter_lon is not None:
        args.epicenter = [args.epicenter_lat, args.epicenter_lon]
    else:
        args.epicenter = None
    if args.xmin is None:
        args.xmin = 0.0
    if args.xmax is None:
        args.xmax = 30.0

    return args

def plot_timeseries_main():
    """
    Main function for plot_timeseries
    """
    # Parse command-line options
    args = parse_arguments()
    # Copy inputs
    output_file = args.outfile
    filenames = args.input_files

    # Set plot title
    plot_title = None
    if args.station is not None:
        plot_title = "%s" % (args.station)
    # Set title if station name provided and epicenter are provided
    if args.station is not None and args.epicenter is not None:
        # Calculate distance if locations are provided
        if args.st_loc is None and args.station_list is not None:
            # Find station coordinates from station list
            st_file = open(args.station_list, 'r')
            for line in st_file:
                line = line.strip()
                if not line:
                    # skip blank lines
                    continue
                if line.startswith("#") or line.startswith("%"):
                    # Skip comments
                    continue
                pieces = line.split()
                if len(pieces) < 3:
                    # Skip line with insufficient tokens
                    continue
                if pieces[2].lower() != args.station.lower():
                    # Not a match
                    continue
                # Match!
                args.st_loc = [float(pieces[1]), float(pieces[0])]
                break
            # All done processing station file
            st_file.close()

        if args.st_loc is not None:
            # Calculate distance here
            distance = calculate_distance(args.epicenter, args.st_loc)
            # Set plot title
            plot_title = "%s, Dist: ~%dkm" % (args.station,
                                              distance)

    # Read data
    _, stations = read_files(None, filenames)
    filenames = [os.path.basename(filename) for filename in filenames]

    # Create plot
    plot_overlay_timeseries(args, filenames, stations,
                            output_file, plot_title=plot_title)

# ============================ MAIN ==============================
if __name__ == "__main__":
    plot_timeseries_main()
# end of main program
