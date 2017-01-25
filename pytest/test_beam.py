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

#beam = hp.Beam(file="TAEM01_input_beam_64K.dat")
#beam.set_frequency(16.6);
beam = hp.Beam(1, 1, 1, 20)
print beam.get_frequency()
beam.set_frequency(16.6)
print beam.get_frequency()
beam.set_frequency(201)
print beam.get_frequency()

