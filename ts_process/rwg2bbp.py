#!/usr/bin/env python
"""
Utility to convert RWG time history files to BBP format
"""
from __future__ import division, print_function

# Import python modules
import os
import sys
import argparse
import numpy as np
from ts_library import integrate, derivative

def get_dt(input_file):
    """
    Read RWG file and return DT
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
        # Skip negative data points
        if pieces[0] < 0.0:
            continue
        if val1 is None:
            val1 = pieces[0]
            continue
        if val2 is None:
            val2 = pieces[0]
            break
    ifile.close()

    # Quit if cannot figure out dt
    if val1 is None or val2 is None:
        print("Cannot determine dt from RWG file! Exiting...")
        sys.exit(1)

    # Calculate dt
    file_dt = val2 - val1

    return file_dt
# end get_dt

def read_rwg(input_file):
    """
    Reads the input file in rwg format and returns arrays containing
    vel_ns, vel_ew, vel_ud components
    """
    original_header = []
    time = np.array([])
    vel_ns = np.array([])
    vel_ew = np.array([])
    vel_ud = np.array([])

    # Get RWG file dt
    delta_t = get_dt(input_file)

    try:
        input_fp = open(input_file, 'r')
        for line in input_fp:
            line = line.strip()
            if line.startswith("#") or line.startswith("%"):
                # keep original header
                original_header.append(line)
                continue
            pieces = line.split()
            pieces = [float(piece) for piece in pieces]
            # Skip negative data points
            if pieces[0] < 0.0:
                continue
            # Add values to out arrays
            time = np.append(time, pieces[0])
            vel_ns = np.append(vel_ns, pieces[1])
            vel_ew = np.append(vel_ew, pieces[2])
            vel_ud = np.append(vel_ud, pieces[3])
    except IOError as e:
        print(e)
        sys.exit(1)

    # All done
    input_fp.close()

    return original_header, delta_t, time, vel_ns, vel_ew, vel_ud

def parse_rwg_header(header):
    """
    Parse the header found in the rwg file to extract useful metadata
    """
    params = {}

    for line in header:
        pieces = line.split()
        if line.find("Station:") > 0:
            params["station"] = pieces[2]
            continue
        if line.find("lon=") > 0:
            params["lon"] = float(pieces[2])
            continue
        if line.find("lat=") > 0:
            params["lat"] = float(pieces[2])
            continue
        if line.find("Column 2:") > 0:
            if pieces[7] == "(m/s)":
                params["unit"] = "m"
            elif pieces[7] == "(cm/s)":
                params["unit"] = "cm"
            continue

    # Return what we found
    return params

def write_bbp_header(out_fp, file_type, file_unit, args, header):
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
    # Now copy precious header lines
    for line in header:
        out_fp.write("#%s\n" % (line))

def rwg2bbp_main():
    """
    Script to convert RWG files to BBP format
    """
    parser = argparse.ArgumentParser(description="Converts an RWG "
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

    # Read RWG file
    header, delta_t, times, vel_h1, vel_h2, vel_ver = read_rwg(input_file)
    rwg_params = parse_rwg_header(header)

    # Figure out what unit to use
    units = {"m": ["m", "m/s", "m/s^2"],
             "cm": ["cm", "cm/s", "cm/s^2"]}
    if "unit" in rwg_params:
        unit = rwg_params["unit"]
    else:
        # Defaults to meters
        unit = "m"

    # Override command-line defaults
    if "lat" in rwg_params:
        if args.latitude == 0.0:
            args.latitude = rwg_params["lat"]
    if "lon" in rwg_params:
        if args.longitude == 0.0:
            args.longitude = rwg_params["lon"]
    if "station" in rwg_params:
        if args.station_name == "NoName":
            args.station_name = rwg_params["station"]

    # Calculate displacement
    dis_h1 = integrate(vel_h1, delta_t)
    dis_h2 = integrate(vel_h2, delta_t)
    dis_ver = integrate(vel_ver, delta_t)

    # Calculate acceleration
    acc_h1 = derivative(vel_h1, delta_t)
    acc_h2 = derivative(vel_h2, delta_t)
    acc_ver = derivative(vel_ver, delta_t)

    # Write header
    o_dis_file = open(output_file_dis, 'w')
    o_vel_file = open(output_file_vel, 'w')
    o_acc_file = open(output_file_acc, 'w')
    write_bbp_header(o_dis_file, "displacement",
                     units[unit][0], args, header)
    write_bbp_header(o_vel_file, "velocity",
                     units[unit][1], args, header)
    write_bbp_header(o_acc_file, "acceleration",
                     units[unit][2], args, header)

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
    rwg2bbp_main()
# end of main program
