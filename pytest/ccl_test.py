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

#@profile
#def prof(beam):
#  option = "good"
#  t0 = time.time()
#  #for i in range(100):
#  x = np.array(beam.get_yp_old(option))
#  t1 = time.time()
#  #for i in range(100):
#  x_np = beam.get_yp(option)
#  t2 = time.time()
#  #print x
#  #print x.shape
#  #print x_np
#  #print x_np.shape
#  print np.max(x-x_np)
#  print (t1-t0)/(t2-t1)
#
#  print np.max(np.abs(np.array(beam.get_w_old(option)) - beam.get_w(option)))
#  print np.max(np.abs(np.array(beam.get_phi_old(option, 'absolute')) - beam.get_phi(option, 'absolute')))
#  print np.max(np.abs(np.array(beam.get_phi_old(option, 'relative')) - beam.get_phi(option, 'relative')))
#  print np.max(np.abs(np.array(beam.get_losses_old('t')) - beam.get_losses('t')));
#  print np.max(np.abs(np.array(beam.get_losses_old('l')) - beam.get_losses('l')));
 
beam = hp.Beam(file="Hm_clz_64k.txt")
conn = hp.DBConnection("../db/offline-ccl.db")
bl = hp.BeamLine(conn)
sim = hp.Simulator(beam=beam, beamline=bl)
sim.simulate('', '')
beam.print_to('end.txt')

#prof(beam)

