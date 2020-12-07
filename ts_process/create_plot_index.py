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

The program creates a html index for a plot folder
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import glob
import argparse
from ts_library import calculate_distance

def parse_arguments():
    """
    This function takes care of parsing the command-line arguments and
    asking the user for any missing parameters that we need
    """
    parser = argparse.ArgumentParser(description="Creates a html index for "
                                     " a direcory of plots.")
    parser.add_argument("-o", "--output", dest="outfile", required=True,
                        help="output html file")
    parser.add_argument("--plotdir", dest="plotdir", required=True,
                        help="directory containing plots")
    parser.add_argument("--epicenter-lat", dest="epicenter_lat", type=float,
                        help="earthquake epicenter latitude")
    parser.add_argument("--epicenter-lon", dest="epicenter_lon", type=float,
                        help="earthquake epicenter longitude")
    parser.add_argument("--station-list", dest="station_list", required=True,
                        help="station list with latitude and longitude")
    parser.add_argument("--freqs", dest="freqs",
                        help="frequencies used for the simulation")
    parser.add_argument("--alpha", dest="alpha",
                        default=False, action='store_true',
                        help="sort output alphabetically")
    parser.add_argument("--title", dest="title",
                        help="title for the html file")
    args = parser.parse_args()

    if args.epicenter_lat is not None and args.epicenter_lon is not None:
        args.epicenter = [args.epicenter_lat, args.epicenter_lon]
    else:
        args.epicenter = None
    if args.freqs is not None:
        args.freqs = args.freqs.strip().split(",")
        args.freqs = [freq.strip() for freq in args.freqs]
    if args.alpha:
        args.order = "alpha"
    else:
        args.order = "distance"
    if args.title is None:
        args.title = "Results"

    return args

def calculate_distances(station_list, epicenter):
    """
    Calculates distanes from all stations
    """
    distances = {}
    
    # Find station coordinates from station list
    st_file = open(station_list, 'r')
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
        station_id = pieces[2].upper()
        station_loc = [float(pieces[1]), float(pieces[0])]
        # Calculate distance here
        station_dist = calculate_distance(epicenter, station_loc)
        distances[station_id] = station_dist
    
    st_file.close()

    return distances

def create_plot_index_main():
    """
    Main function for create_plot_index
    """
    # Parse command-line options
    args = parse_arguments()
    # Copy inputs
    output_file = args.outfile
    plots_dir = args.plotdir
    freqs = args.freqs
    
    # Calculate distances from epicenter
    distances = calculate_distances(args.station_list, args.epicenter)

    # Create ordered lists
    alpha_sort = sorted(distances)
    dist_sort2 = sorted(distances.items(), key=lambda item: item[1])
    dist_sort = [item for item, _ in dist_sort2]

    if args.order == "distance":
        station_order = dist_sort
    else:
        station_order = alpha_sort

    # Open html file
    html_output = open(output_file, 'w')
    html_output.write("<html>\n")
    html_output.write("<title>%s</title>\n" % (args.title))
    html_output.write("<body>\n")
    html_output.write("<h2>%s</h2>\n" % (args.title))
    html_output.write("<table>\n")
        
    for station in station_order:
        files = glob.glob("%s/%s*" % (plots_dir, station))
        # Skip stations that don't have any plots
        if not files:
            continue
        html_output.write("<tr>\n")
        html_output.write("<td>%s</td>\n" % (station))
        html_output.write("<td>%f</td>\n" % (distances[station]))
        for freq in args.freqs:
            files = glob.glob("%s/%s-%s*" % (plots_dir, station, freq))
            if len(files) != 1:
                continue
            html_output.write('<td><a href="%s">%s</a></td>\n' %
                              (os.path.basename(files[0]), freq))
        html_output.write("</tr>\n")

    html_output.write("</table>\n")
    html_output.write("</body>\n")
    html_output.write("</html>\n")
    html_output.close()
        
# ============================ MAIN ==============================
if __name__ == "__main__":
    create_plot_index_main()
# end of main program
