#!/usr/bin/env python
# sim-lbeg-scan-pv.py
# simulate lbeg H- beam through the LANSCE beamline while varying EPICS PV over range of vals

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
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
# import additional simulation packages
import hpsim as hps
import HPSim as HPSim
# use next line to select either GPU 0 or 1
hps.set_gpu(2)
import lcsutil as lcs
import nputil as npu

################################################################################
# install db's and connect to beamline
db_dir = par_dir + '/db'
lib_dir = par_dir + '/db/lib'
dbs = ['tbtd.db','dtl.db','trst.db','ccl.db']
dbconn1 = hps.DBConnection(db_dir, dbs, lib_dir, 'libsqliteext.so')
dbconn1.print_dbs()
dbconn1.clear_model_index()
print "*** dB connection established ***"

################################################################################
# create beamline
bl = hps.BeamLine()
beamline = hps.get_element_list()
print "*** Beamline created ***"

################################################################################
# create H- beam
SIM_START = "TBDB02" #defined by input beam location
beam = hps.Beam(mass=939.294, charge=-1.0, current=0.015, num=1024*64) #H- beam
beam.set_dc(0.095, 47.0, 0.00327,  -0.102, 60.0, 0.002514, 180.0, 0.0, 0.7518) #TBDB02 20140901
beam.set_frequency(201.25)
betalambda = hps.betalambda(mass = beam.get_mass(), freq=beam.get_frequency(), w=0.750)
phi_offset = -hps.get_beamline_length('TBDB02','BLZ')/betalambda *360
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
#spch.set_remesh_threshold(0.02)
spch.set_remesh_threshold(0.2)
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

SIM_STOP = 'TREM01'
ENERGY_CUTOFF = 95.0 #MeV; should be less than the nominal beam output energy
PV_TYPE = 'AMP' #'AMP' or 'PHS' or 'OTHER' or 'NONE'
AREA = 'TD'
BEAM = '-'

if PV_TYPE == 'OTHER':
    # if OTHER then define EPICS_PV here and adjust scan range below
    EPICS_PV = 'MRPH001D01' #master reference phase for whole CCL

if PV_TYPE == None:
    EPICS_PV = None

if PV_TYPE == 'PHS':
# EPICS pv to scan
    EPICS_PV = lcs.get_pv_psp(AREA, beam=BEAM) 
    # Initial pv PHS value; will restore after scan
    PV_INIT_VAL = hps.get_db_epics(EPICS_PV)
    D_PV = 20.0 #degrees
    PV_MIN =PV_INIT_VAL - D_PV
    PV_MAX = PV_INIT_VAL + D_PV
    PV_STEP = 5.0 #degree

elif PV_TYPE == 'AMP':
    EPICS_PV = lcs.get_pv_asp(AREA, beam=BEAM) 
    # Initial pv AMP value; will restore after scan
    PV_INIT_VAL = hps.get_db_epics(EPICS_PV)
    D_PV = 5.0 #percent of initial value
    PV_MIN = PV_INIT_VAL * (1.0 - D_PV/100.)
    PV_MAX = PV_INIT_VAL * (1.0 + D_PV/100.)
    STEP_SIZE = 2.5 #percent of initial value
    PV_STEP = STEP_SIZE/100.0 * PV_INIT_VAL
    print np.arange(PV_MIN, PV_MAX, PV_STEP)

elif PV_TYPE == 'OTHER':
    PV_INIT_VAL = hps.get_db_epics(EPICS_PV)
    D_PV = 120.
    PV_MIN = PV_INIT_VAL - D_PV 
    PV_MAX = PV_INIT_VAL + D_PV 
    PV_STEP = 10.0 

elif PV_TYPE == None:
    EPICS_PV = None
    print PV_TYPE

else:
    print 'Error with pv_type'
    exit()

print '{0} starting value is {1}'.format(EPICS_PV, PV_INIT_VAL)

try: #use try to allow graceful exit if simulation crashes, so that PV is restored to init val
    plt.ion() #interactive mode ON
    plot = hps.DistPlot(nrow=4, ncol=3, hsize=16, vsize=12)
    output = []
    bll = []

    for val in np.arange(PV_MIN, PV_MAX, PV_STEP):
        beam.restore_initial_beam()
        hps.set_db_epics(EPICS_PV, val)
        print EPICS_PV, "is now", hps.get_db_epics(EPICS_PV)
    # simulate here
        sim.simulate(SIM_START, SIM_STOP)
        
        wmask = beam.get_mask_with_limits('w',lolim=ENERGY_CUTOFF)
        mask = beam.get_good_mask(wmask)

        if len(mask) > 0:
            try:
                dist = hps.Distribution(beam, mask)
            # beam.print_results(mask)
            # save npart, avgW, sigW, avgPHI, sigPHI beam quantities to output list
                output.append([val, dist.get_size(), 
                               dist.get_avg('w'), \
                               dist.get_sig('w'), \
                               dist.get_avg('phi'), \
                               dist.get_sig('phi')])
            
                plot.clear()
                # create intermediate results to plot for each scan step 
                plot.iso_phase_space('xxp', dist, 1)
                plot.iso_phase_space('yyp', dist, 2)
                plot.iso_phase_space('phiw', dist, 3)
                plot.hist2d_phase_space('xxp', dist, 4)
                plot.hist2d_phase_space('yyp', dist, 5)
                plot.hist2d_phase_space('phiw', dist, 6)
                plot.profile('x', dist, 7, 'r-')
                plot.profile('y', dist, 8, 'r-')
                plot.profile('phi', dist, 9, 'r-')
                plot.profile('xp', dist, 10, 'r-')
                plot.profile('yp', dist, 11, 'r-')
                plot.profile('w', dist, 12, 'r-')
                title = "H{0} from {1} to {2}; {3} {4} = {5}".format(\
                    BEAM, SIM_START, SIM_STOP, PV_TYPE, EPICS_PV, val)
                plot.title(title)
                plot.draw()
            except:
                print " Warning - No output for PV at this value"
finally:
    plt.ioff()
    hps.set_db_epics(EPICS_PV, PV_INIT_VAL)
    print '{0} restore to original value {1}'.format(EPICS_PV, PV_INIT_VAL)

    # print results
    for item in output:
        print item

# plot output list quantities
    pv_val, npart, wout, sig_w, avg_phi, sig_phi = zip(*output)
        
    fig2 = plt.figure()
    title = "H{0} from {1} to {2}; scan of {3} {4}".format(\
        BEAM, SIM_START, SIM_STOP, PV_TYPE, EPICS_PV)
    fig2.canvas.set_window_title(title)
    
    a1 = fig2.add_subplot(511)
    a1.plot(pv_val, npart, 'b-')
    a1.set_ylabel('Num part')
    npm = max(npart)
    a1.set_ylim([0.95*npm, 1.05*npm])
    
    a2 = fig2.add_subplot(512)
    a2.plot(pv_val, wout, 'b-')
    a2.set_ylabel('W out (Mev)')
    
    a3 = fig2.add_subplot(513)
    a3.plot(pv_val, sig_w, 'b-')
    a3.set_ylabel('Sigma W (MeV)')
    
    a4 = fig2.add_subplot(514)
    a4.plot(pv_val, avg_phi, 'b-')
    a4.set_ylabel('Avg Phi (deg)')

    a5 = fig2.add_subplot(515)
    a5.plot(pv_val, sig_phi, 'b-')
    a5.set_ylabel('Sigma phi (deg)')    
    a5.set_xlabel(EPICS_PV)
    
    plt.show()
exit()
