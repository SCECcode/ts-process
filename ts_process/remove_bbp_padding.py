#!/usr/bin/env python
"""
Copyright 2010-2020 University Of Southern California

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module removes the padding added to a BBP file.
"""
from __future__ import division, print_function

# Import Python modules
import os
import sys
import glob
import shutil
import argparse
from file_utilities import get_dt, read_padding_bbp

def parse_arguments():
    """
    Parse command-line options
    """
    parser = argparse.ArgumentParser(description="Remove padding "
                                     "from a set of BBP seismograms.")
    parser.add_argument("--input_dir", "-i", dest="input_dir",
                        required=True, help="input directory")
    parser.add_argument("--output_dir", "-o", dest="output_dir",
                        required=True, help="output directory")
    parser.add_argument("--prefix", "-p", dest="prefix",
                        default="",
                        help="prefix for input files")
    parser.add_argument("--suffix", "-s", dest="suffix",
                        default="",
                        help="suffix for input files")
    args = parser.parse_args()

    return args

def remove_padding(input_file, output_file, padding):
    """
    Remove padding from BBP file
    """
    # Read DT
    dt = get_dt(input_file)
    current_time = 0.0
    total_points = 0
    current_point = 0

    # First need to figure out how many datapoints we have
    in_fp = open(input_file, 'r')
    for line in in_fp:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("%"):
            continue
        total_points = total_points + 1
    in_fp.close()

    end_point = total_points - padding
    print("total: %d, padding: %d, end: %d" % (total_points, padding, end_point))

    in_fp = open(input_file, 'r')
    out_fp = open(output_file, 'w')
    for line in in_fp:
        line = line.strip()
        if not line:
            out_fp.write("\n")
            continue
        if line.startswith("#") or line.startswith("%"):
            # Header, copy but skip padding
            if line.find("padding=") > 0:
                out_fp.write("#     padding= 0\n")
                continue
            out_fp.write("%s\n" % (line))
            continue

        # Check if we are done
        if current_point == end_point:
            break

        # Keep track of points
        current_point = current_point + 1

        # Actual data
        if padding > 0:
            # Skip this point
            padding = padding - 1
            continue

        # Use this point
        tokens = line.split()
        tokens = [float(token) for token in tokens]
        out_fp.write("%5.7f   %5.9e   %5.9e   %5.9e\n" %
                     (current_time, tokens[1], tokens[2], tokens[3]))
        # Advance time
        current_time = current_time + dt

    in_fp.close()
    out_fp.close()

def bbp_remove_padding():
    """
    Create a set of BBP files without padding
    """
    # Get all we need from the command-line
    args = parse_arguments()

    # Get list of matching input files
    files = glob.glob("%s/%s*%s" % (args.input_dir, args.prefix, args.suffix))

    for input_file in sorted(files):
        print("[PROCESSING]: %s" % (os.path.basename(input_file)))
        input_base = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, input_base)
        padding = read_padding_bbp(input_file)
        if padding > 0:
            # Found padding that needs to be removed
            print("[INFO]: Found padding %d..." % (padding))
            remove_padding(input_file, output_file, padding)
        else:
            print("[COPYING]: Found no padding, copying file...")
            shutil.copy2(input_file, output_file)

if __name__ == '__main__':
    bbp_remove_padding()
