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

The program contains several input/output utility
functions used by other modules.
"""
from __future__ import division, print_function, absolute_import

# Import Python modules
import os
import sys
import numpy as np

# Import seismtools needed classes
from ts_library import TimeseriesComponent

def reverse_up_down(station):
    """
    reverse up down component
    """
    # station has 3 components [ns, ew, ud]
    # only need to flip the 3rd one
    station[2].acc *= -1
    station[2].vel *= -1
    station[2].dis *= -1

    return station
# end of reverse_up_down

def scale_from_m_to_cm(station):
    # scales timeseries from meters to centimeters
    for i in range(0, len(station)):
        station[i].acc *= 100
        station[i].vel *= 100
        station[i].dis *= 100

    return station
# end of scale_from_m_to_cm

def get_dt(input_file):
    """
    Read timeseries file and return dt
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
        print("[ERROR]: Cannot determine dt from file! Exiting...")
        sys.exit(1)

    # Return dt
    return val2 - val1
# end get_dt

def read_files(obs_file, input_files):
    """
    Reads all input files
    """
    # read obs data
    obs_data = None
    if obs_file is not None:
        obs_data = read_file(obs_file)
        # Make sure we got it
        if not obs_data:
            print("[ERROR]: Reading obs file: %s!" % (obs_file))
            sys.exit(-1)
        # Fix units if needed
        if obs_file.lower().endswith(".bbp"):
            units = read_unit_bbp(obs_file)
            # If in meters, scale to cm
            if units == "m":
                obs_data = scale_from_m_to_cm(obs_data)
        else:
            print("[ERROR]: Unknown file format: %s!" % (obs_file))
            sys.exit(-1)

    # reads signals
    stations = []
    for input_file in input_files:
        station = read_file(input_file)
        # Make sure we got it
        if not station:
            print("[ERROR]: Reading input file: %s!" % (input_file))
            sys.exit(-1)
        # Fix units if needed
        if input_file.lower().endswith(".bbp"):
            units = read_unit_bbp(input_file)
            # If in meters, scale to cm
            if units == "m":
                station = scale_from_m_to_cm(station)
        else:
            print("[ERROR]: Unknown file format: %s!" % (obs_file))
            sys.exit(-1)

        # Done with this station
        stations.append(station)

    # all done
    return obs_data, stations

def read_filelist(filelist):
    """
    This function reads the filelist provided by the user
    """
    station_list = []
    coor_x = []
    coor_y = []

    try:
        input_file = open(filelist, 'r')
    except IOError:
        print("[ERROR]: error loading filelist.")
        sys.exit(-1)

    for line in input_file:
        if not '#' in line:
            line = line.split()
            # Get station name and make substitution
            station_name = line[0]
            station_name = station_name.replace(".", "_")

            if len(line) == 1:
                # not containing coordinates
                station_list.append(station_name)
                coor_x.append(0.0)
                coor_y.append(0.0)
            elif len(line) == 3:
                # containing coordinates
                station_list.append(station_name)
                try:
                    coor_x.append(float(line[1]))
                    coor_y.append(float(line[2]))
                except ValueError:
                    coor_x.append(0.0)
                    coor_y.append(0.0)

    # Close the input file
    input_file.close()

    return station_list, coor_x, coor_y
# end of read_filelist

# ================================ READING ================================
def read_file(filename):
    """
    This function reads a timeseries file in bbp format
    """
    if filename.lower().endswith(".bbp"):
        # Filename in bbp format
        print("[READING]: %s" % (filename))
        return read_file_bbp(filename)
    # Unknown file format
    print("[ERROR]: Unknown file format: %s!" % (obs_file))
    sys.exit(-1)
# end of read_file

def read_file_bbp2(filename):
    """
    This function reads a bbp file and returns the timeseries in the
    format time, h1, h2, up tuple
    """
    time = []
    h1_comp = []
    h2_comp = []
    ud_comp = []

    try:
        input_file = open(filename, 'r')
        for line in input_file:
            line = line.strip()
            if line.startswith('#') or line.startswith('%'):
                # Skip comments
                continue
            # Trim in-line comments
            if line.find('#') > 0:
                line = line[:line.find('#')]
            if line.find('%') > 0:
                line = line[:line.find('%')]
            # Make them float
            pieces = line.split()
            pieces = [float(piece) for piece in pieces]
            time.append(pieces[0])
            h1_comp.append(pieces[1])
            h2_comp.append(pieces[2])
            ud_comp.append(pieces[3])
    except IOError:
        print("[ERROR]: error reading bbp file: %s" % (filename))
        sys.exit(1)

    # Convert to NumPy Arrays
    time = np.array(time)
    h1_comp = np.array(h1_comp)
    h2_comp = np.array(h2_comp)
    ud_comp = np.array(ud_comp)

    # All done!
    return time, h1_comp, h2_comp, ud_comp
# end of read_file_bbp2

def read_file_bbp(filename):
    """
    This function reads timeseries data from a set of BBP files
    """
    # Get filenames for displacement, velocity and acceleration bbp files
    work_dir = os.path.dirname(filename)
    base_file = os.path.basename(filename)

    base_tokens = base_file.split('.')[0:-2]
    if not base_tokens:
        print("[ERROR]: Invalid BBP filename: %s" % (filename))
        sys.exit(1)
    dis_tokens = list(base_tokens)
    vel_tokens = list(base_tokens)
    acc_tokens = list(base_tokens)

    dis_tokens.append('dis')
    vel_tokens.append('vel')
    acc_tokens.append('acc')

    dis_tokens.append('bbp')
    vel_tokens.append('bbp')
    acc_tokens.append('bbp')

    dis_file = os.path.join(work_dir, '.'.join(dis_tokens))
    vel_file = os.path.join(work_dir, '.'.join(vel_tokens))
    acc_file = os.path.join(work_dir, '.'.join(acc_tokens))

    # Read 3 bbp files
    [time, dis_h1, dis_h2, dis_ver] = read_file_bbp2(dis_file)
    [_, vel_h1, vel_h2, vel_ver] = read_file_bbp2(vel_file)
    [_, acc_h1, acc_h2, acc_ver] = read_file_bbp2(acc_file)

    # Read orientation from one of the files
    orientation = read_orientation_bbp(vel_file)

    # Read padding information from one of the files
    padding = read_padding_bbp(vel_file)

    samples = dis_h1.size
    delta_t = time[1]

    # samples, dt, data, acceleration, velocity, displacement
    signal_h1 = TimeseriesComponent(samples, delta_t, orientation[0],
                                    acc_h1, vel_h1, dis_h1, padding=padding)
    signal_h2 = TimeseriesComponent(samples, delta_t, orientation[1],
                                    acc_h2, vel_h2, dis_h2, padding=padding)
    signal_ver = TimeseriesComponent(samples, delta_t, orientation[2],
                                     acc_ver, vel_ver, dis_ver, padding=padding)

    station = [signal_h1, signal_h2, signal_ver]
    return station
# end of read_file_bbp

def read_file_her(filename):
    """
    The function is to read 10-column .her files.
    Return a list of psignals for each orientation.
    """
    time, dis_ns, dis_ew, dis_up = [np.array([], float) for _ in xrange(4)]
    vel_ns, vel_ew, vel_up = [np.array([], float) for _ in xrange(3)]
    acc_ns, acc_ew, acc_up = [np.array([], float) for _ in xrange(3)]

    try:
        (time, dis_ns, dis_ew, dis_up, vel_ns, vel_ew,
         vel_up, acc_ns, acc_ew, acc_up) = np.loadtxt(filename,
                                                      comments='#',
                                                      unpack=True)
    except IOError:
        print("[ERROR]: error loading her file.")
        return False

    samples = dis_ns.size
    delta_t = time[1]

    # samples, dt, orientation, acceleration, velocity, displacement
    # right now the values for orientation for the her file are hardcoded here
    signal_ns = TimeseriesComponent(samples, delta_t, 0.0,
                                    acc_ns, vel_ns, dis_ns)
    signal_ew = TimeseriesComponent(samples, delta_t, 90.0,
                                    acc_ew, vel_ew, dis_ew)
    signal_up = TimeseriesComponent(samples, delta_t, "UP",
                                    acc_up, vel_up, dis_up)

    station = [signal_ns, signal_ew, signal_up]
    return station
# end of read_file_her

def read_unit_bbp(filename):
    """
    Get the units from the file's header
    Returns either "m" or "cm"
    """
    units = None

    try:
        input_file = open(filename, 'r')
        for line in input_file:
            if line.find("units=") > 0:
                units = line.split()[2]
                break
        input_file.close()
    except IOError:
        print("[ERROR]: No such file.")
        sys.exit(-1)

    # Make sure we got something
    if units is None:
        print("[ERROR]: Cannot find units in bbp file!")
        sys.exit(-1)

    # Figure out if we have meters or centimeters
    if units == "cm" or units == "cm/s" or units == "cm/s^2":
        return "cm"
    elif units == "m" or units == "m/s" or units == "m/s^2":
        return "m"

    # Invalid units in this file
    print("[ERROR]: Cannot parse units in bbp file!")
    sys.exit(-1)
# end of read_unit_bbp

def read_padding_bbp(filename):
    """
    Get the padding information from a BBP file's header
    """
    padding = 0

    try:
        input_file = open(filename, 'r')
        for line in input_file:
            if line.find("padding=") > 0:
                line = line.strip()
                padding = line[(line.find("=") + 1):]
                padding = int(float(padding))
                break
        input_file.close()
    except IOError:
        print("[ERROR]: No such file.")
        sys.exit(-1)

    # All done!
    return padding
# end of read_padding_bbp

def read_orientation_bbp(filename):
    """
    Get the orientation from the file's header
    """
    orientation = None

    try:
        input_file = open(filename, 'r')
        for line in input_file:
            if line.find("orientation=") > 0:
                line = line.strip()
                orientation = line[(line.find("=") + 1):]
                orientation = orientation.strip().split(",")
                orientation = [val.strip() for val in orientation]
                orientation[0] = float(orientation[0])
                orientation[1] = float(orientation[1])
                orientation[2] = orientation[2].lower()
                if orientation[2] != "up" and orientation[2] != "down":
                    print("[ERROR]: Vertical orientation must be up or down!")
                    sys.exit(-1)
                break
        input_file.close()
    except IOError:
        print("[ERROR]: No such file.")
        sys.exit(-1)

    # Make sure we got something
    if orientation is None:
        print("[ERROR]: Cannot find orientation in bbp file: %s!" % (filename))
        sys.exit(-1)

    # All done!
    return orientation
# end of read_orientation_bbp

def read_stamp(filename):
    """
    Get the time stamp from file's header
    """
    if filename.endswith(".bbp"):
        # File in bbp format
        return read_stamp_bbp(filename)
    # Otherwise use hercules format
    return read_stamp_her(filename)
# end of read_stamp

def read_stamp_bbp(filename):
    """
    Get the time stamp from the bbp file's header
    """
    try:
        input_file = open(filename, 'r')
        for line in input_file:
            if line.find("time=") > 0:
                stamp = line.split()[2].split(',')[-1].split(':')
                break
        input_file.close()
    except IOError:
        print("[ERROR]: No such file.")
        return []

    # Converting time stamps to floats
    stamp = [float(i) for i in stamp]
    return stamp
# end of read_stamp_bbp

def read_stamp_her(filename):
    """
    Get the time stamp from the her file's header
    """
    try:
        with open(filename) as input_file:
            try:
                header = input_file.readline().split()
                stamp = header[4].split(',')[-1].split(':')
                input_file.close()
            except IndexError:
                print("[ERROR]: missing time stamp.")
                return []
    except IOError:
        print("[ERROR]: No such file.")
        return []

    # converting time stamps to floats
    for i in range(0, len(stamp)):
        stamp[i] = float(stamp[i])
    return stamp
# end of read_stamp_her

# ================================ WRITING ==================================
def write_hercules(filename, station):
    # filename = 'processed-' + filename.split('/')[-1]
    try:
        out_f = open(filename, 'w')
    except IOError as e:
        print(e)
    dis_ns = station[0].dis.tolist()
    vel_ns = station[0].vel.tolist()
    acc_ns = station[0].acc.tolist()
    dis_ew = station[1].dis.tolist()
    vel_ew = station[1].vel.tolist()
    acc_ew = station[1].acc.tolist()
    dis_up = station[2].dis.tolist()
    vel_up = station[2].vel.tolist()
    acc_up = station[2].acc.tolist()

    # get a list of time incremented by dt
    time = [0.000]
    samples = station[0].samples
    dt = station[0].dt
    tmp = samples

    while tmp > 1:
        time.append(time[len(time)-1] + dt)
        tmp -= 1

    out_f.write('# missing header \n')

    descriptor = '{:>12}' + '  {:>12}'*9 + '\n'
    out_f.write(descriptor.format("# time",
                                  "dis_ns", "dis_ew", "dis_up",
                                  "vel_ns", "vel_ew", "vel_up",
                                  "acc_ns", "acc_ew", "acc_up")) # header

    descriptor = '{:>12.3f}' + '  {:>12.7f}'*9 + '\n'
    for c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 in zip(time,
                                                      dis_ns, dis_ew, dis_up,
                                                      vel_ns, vel_ew, vel_up,
                                                      acc_ns, acc_ew, acc_up):
        out_f.write(descriptor.format(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9))
    out_f.close()
# end of write_hercules

def write_bbp(input_file, output_file, station, params={}):
    """
    This function generates processed .bbp files for
    each of velocity/acceleration/displacement
    and copies the header of the input bbp file
    """
    output_dir = os.path.dirname(output_file)
    output_basename = os.path.basename(output_file)

    # Prepare data for output
    acc_h1 = station[0].acc.tolist()
    vel_h1 = station[0].vel.tolist()
    dis_h1 = station[0].dis.tolist()
    acc_h2 = station[1].acc.tolist()
    vel_h2 = station[1].vel.tolist()
    dis_h2 = station[1].dis.tolist()
    acc_ver = station[2].acc.tolist()
    vel_ver = station[2].vel.tolist()
    dis_ver = station[2].dis.tolist()

    # Start with time = 0.0
    time = [0.000]
    samples = station[0].samples
    while samples > 1:
        time.append(time[len(time)-1] + station[0].dt)
        samples -= 1

    # Prepare to output
    out_data = [['dis', dis_h1, dis_h2, dis_ver, 'displacement', 'cm'],
                ['vel', vel_h1, vel_h2, vel_ver, 'velocity', 'cm/s'],
                ['acc', acc_h1, acc_h2, acc_ver, 'acceleration', 'cm/s^2']]

    for data in out_data:
        if not output_basename.endswith('.bbp'):
            # Remove extension
            bbp_output_basename = os.path.splitext(output_basename)[0]
            bbp_output_filename = os.path.join(output_dir,
                                               "%s.%s.bbp" %
                                               (bbp_output_basename,
                                                data[0]))
            output_header = ["#     Station= NoName",
                             "#        time= 00/00/00,00:00:00.00 UTC",
                             "#         lon= 0.00",
                             "#         lat= 0.00",
                             "#       units= %s" % (data[5]),
                             "#     padding= %d" % (station[0].padding),
                             "# orientation= %s" % (",".join([str(int(station[0].orientation)),
                                                              str(int(station[1].orientation)),
                                                              station[2].orientation])),
                             "#",
                             "# Data fields are TAB-separated",
                             "# Column 1: Time (s)",
                             "# Column 2: H1 component ground "
                             "%s (+ is %s)" % (data[4],
                                               str(int(station[0].orientation))),
                             "# Column 3: H2 component ground "
                             "%s (+ is %s)" % (data[4],
                                               str(int(station[1].orientation))),
                             "# Column 4: V component ground "
                             "%s (+ is %s)" % (data[4], station[2].orientation),
                             "#"]
        else:
            # Read header of input file
            input_dirname = os.path.dirname(input_file)
            input_basename = os.path.basename(input_file)
            pieces = input_basename.split('.')
            pieces = pieces[0:-2]
            bbp_input_file = os.path.join(input_dirname,
                                          "%s.%s.bbp" %
                                          ('.'.join(pieces),
                                           data[0]))
            input_header = []
            in_fp = open(bbp_input_file, 'r')
            for line in in_fp:
                line = line.strip()
                if line.startswith("#"):
                    input_header.append(line)
            in_fp.close()

            # Compose new header
            output_header = []
            for item in input_header:
                if item.find("units=") > 0:
                    output_header.append("#       units= %s" % (data[5]))
                elif item.find("orientation=") > 0:
                    output_header.append("# orientation= %s" % (",".join([str(int(station[0].orientation)),
                                                                          str(int(station[1].orientation)),
                                                                          station[2].orientation])))
                elif item.find("lp=") > 0:
                    if 'lp' in params and params['lp'] is not None:
                        output_header.append("#          lp= %.2f" % (params['lp']))
                    else:
                        output_header.append(item)
                elif item.find("hp=") > 0:
                    if 'hp' in params and params['hp'] is not None:
                        output_header.append("#          hp= %.2f" % (params['hp']))
                    else:
                        output_header.append(item)
                elif item.find("padding=") > 0:
                    output_header.append("#     padding= %d" % (station[0].padding))
                elif item.find("Column 2") > 0:
                    output_header.append("# Column 2: H1 component ground "
                                         "%s (+ is %s)" % (data[4],
                                                           str(int(station[0].orientation))))
                elif item.find("Column 3") > 0:
                    output_header.append("# Column 3: H2 component ground "
                                         "%s (+ is %s)" % (data[4],
                                                           str(int(station[1].orientation))))
                elif item.find("Column 4") > 0:
                    output_header.append("# Column 4: V component ground "
                                         "%s (+ is %s)" % (data[4], station[2].orientation))
                else:
                    output_header.append(item)

            pieces = output_basename.split('.')
            pieces = pieces[0:-2]
            bbp_output_filename = os.path.join(output_dir,
                                               "%s.%s.bbp" %
                                               ('.'.join(pieces),
                                                data[0]))
        # Write output file
        try:
            out_fp = open(bbp_output_filename, 'w')
        except IOError as e:
            print(e)
            continue

        # Write header
        for item in output_header:
            out_fp.write("%s\n" % (item))

        # Write timeseries
        for val_time, val_ns, val_ew, val_ud in zip(time, data[1],
                                                    data[2], data[3]):
            out_fp.write("%5.7f   %5.9e   %5.9e    %5.9e\n" %
                         (val_time, val_ns, val_ew, val_ud))

        # All done, close file
        out_fp.close()
        print("[WRITING]: %s" % (bbp_output_filename))
# end of write_bbp
