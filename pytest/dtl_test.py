#!/usr/local/bin/python 
import os
import sys
import numpy as np
import time

par_dir = os.path.abspath(os.path.pardir)
lib_dir = os.path.join(par_dir, "bin")
print lib_dir
sys.path.append(lib_dir)

import HPSim as hp

beam = hp.Beam(file="TAEM01_input_beam_64K.dat")
conn = hp.DBConnection("../db/offline-dtl.db")
bl = hp.BeamLine(conn)
sim = hp.Simulator(beam=beam, beamline=bl)
sim.simulate('', '')
beam.print_to('end.txt')
names = bl.get_element_names()
print len(names)
