#!/usr/bin/env python
#
# sim-lbeg.py
# for simulating LBEG beam from point A to B in the LANSCE linac
#
import sys
import os
# define directory to packages and append to $PATH
par_dir = os.path.abspath(os.path.pardir)
print par_dir
lib_dir = os.path.join(par_dir,"bin")
print lib_dir
sys.path.append(lib_dir)
pkg_dir = os.path.join(par_dir,"pylib")
print pkg_dir
sys.path.append(pkg_dir)

#import additional python packages
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import math
# import additional simulation packages
import hpsim as hps
import HPSim as HPSim
# use next line to select either GPU 0 or 2 on aothpsim
GPU = 0
hps.set_gpu(GPU)
import lcsutil as lcs
import nputil as npu
import sqldb as pydb

################################################################################
# install db's and connect to beamline
db_dir = par_dir + '/db'
lib_dir = par_dir + '/db/lib/'
dbs = ['tbtd.db','dtl.db','trst.db','ccl.db']
dbconn1 = hps.DBConnection(db_dir, dbs, lib_dir, 'libsqliteext')
dbconn1.print_libs()
dbconn1.print_dbs()
dbconn1.clear_model_index()
print "*** dB connection established ***"

################################################################################
# create beamline
bl = hps.BeamLine()
beamline = hps.get_element_list()
print "*** Beamline created ***"

################################################################################
# create table of beamline elements at lengths
pybl = pydb.Db_bl(db_dir, dbs)
py_beamline = pybl.get_bl_elem_len()
print "*** PySQLite Beamline created ***"

################################################################################
# create H- beam
SIM_START = "TBDB02" #defined by input beam location
#beam = hps.Beam(mass=939.294, charge=-1.0, current=0.015, num=1024*256) #H- beam
beam = hps.Beam(mass=939.294, charge=-1.0, current=0.015, num=1024*256/4) #H- beam
beam.set_dc(0.095, 47.0, 0.00327,  -0.102, 60.0, 0.002514, 180.0, 0.0, 0.7518) #TBDB02 20140901
beam.set_frequency(201.25)
betalambda = hps.betalambda(mass = beam.get_mass(), freq=beam.get_frequency(), w=0.750)
phi_offset = -hps.get_beamline_length(SIM_START,'BLZ')/betalambda *360
beam.set_ref_w(0.750)
beam.set_ref_phi(phi_offset)
beam.translate('phi', phi_offset)
beam.save_initial_beam()
print "*** H- Beam created ***"

################################################################################
# create spacecharge
spch = hps.SpaceCharge(nr = 32, nz = 128, interval = 0.025, adj_bunch = 3)
print "spch interval=", spch.get_interval()
print "adj_bunch=", spch.get_adj_bunch()
# define at what energy simulation stops using adjacent bunches in SC calc
spch.set_adj_bunch_cutoff_w(0.8)
# remeshing factor determines how ofter the mesh gets recalc vs scaled for SC kick
spch.set_remesh_threshold(0.02)
#spch.set_remesh_threshold(0.2)
print "cutoff w=", spch.get_adj_bunch_cutoff_w()
print "*** Space Charge Initialized ***"

################################################################################
# create simulator
sim = hps.Simulator(beam)
sim.set_space_charge('on')
print "*** Simulator Initialized ***"

################################################################################
# STANDARD AND REQUIRED STUFF ABOVE THIS LINE
################################################################################

SIM_STOP = '48DT'
ENERGY_CUTOFF = 0.0
mask = gmask = beam.get_good_mask()

print "*** Input Beam ***"
print SIM_START
print "w/user units"
beam.print_results()

print "*** Starting Simulation ***\n"
sim.simulate(SIM_START, SIM_STOP)

# determine mask of particles used in analysis and plotting
wmask = beam.get_mask_with_limits('w', lolim = ENERGY_CUTOFF)
gmask = beam.get_good_mask(wmask)
mask = gmask

print "*** Output Beam ***"
print SIM_STOP
print "w/user units"
beam.print_results(mask)

# create output plot
#plot = hps.BeamPlot(nrow=4, ncol=3, hsize=16, vsize=12)
#plot.title(SIM_STOP)
#plot.iso_phase_space('xxp', beam, mask, 1)
#plot.iso_phase_space('yyp', beam, mask, 2)
#plot.iso_phase_space('phiw', beam, mask, 3 )
#plot.hist2d_phase_space('xxp', beam, mask, 4)
#plot.hist2d_phase_space('yyp', beam, mask, 5)
#plot.hist2d_phase_space('phiw', beam, mask, 6)
#plot.profile('x', beam, mask, 7, 'g-')
#plot.profile('y', beam, mask, 8, 'g-')
#plot.profile('phi', beam, mask, 9, 'g-')
#plot.profile('xp', beam, mask, 10, 'g-')
#plot.profile('yp', beam, mask, 11, 'g-')
#plot.profile('w', beam, mask, 12, 'g-')
#plot.show()
exit()
