#ps201.py
#python scripts for creating and manipulating LANSCE PS201 data
import sys
import os
#import additional packages
import numpy as np
from scipy import optimize
import math
import random
if not sys.platform == 'darwin': import hpsim_v6 as hps
import lcsutil_v2 as lcs
import nputil_v2 as npu
from datetime import datetime
# Define global variables for accessing x and y values inside arrays
# Use single _ so class can access them
_X = 0 
_Y = 1

class PS201():
    """A class to represent and analyze the LANSCE PS201 data"""
    def __init__(self):
        """Create an empty instance of PS201 class
	PS201 Class Object Attributes:
        header = first line of file (str)
        date = date & time when scan occurred
        pkcur = peak current (float [mA])
        gatelen = beam gate length (int)
        reprate = repetition rate (int [Hz])
        bgcur = background current (float [mA])
        actuator = Actuator status (list o str)
        target = Nominal Phase of Tank being scanned (used to calc DCLE)
        cavsp = Cavity set poitns (list of list of values)
        npts = number of scan points (int)
        data = Numpy array of phase & collector current pairs (list of floats)
          Additional quantities in class (useful for fitting and simulating)
        plimits = phase limits for analysis purposes (pair of floats)
        threshold = threshold applied to collector current for analysis purposes (float)
        fit_sf = overall scale factor applied to simulated beam current to match measurement
        fit_parms = contains list of model_db fitting parameters for scanned cavity
	"""
        self.header = ""
        self.date = ""
        self.beamgate = ""
        self.cavity = ""
        self.pkcur = ""
        self.gatelen = ""
        self.reprate = ""
        self.date = ""
        self.bgcur = ""
        self.actuator = []
        self.target = ""
        self.cavsp = []
        self.npts = ""
        self.data = []
        self.plimits = []
        self.threshold = 0.0
        self.fit_sf = 1.0
        self.fit_parms = []
# end of __init__

    def init_from_file(self, r_ps201path):
        """Read LANSCE ps201 data located at 'r_lharppath' and store in ps201 object"""
        inpfile = open(r_ps201path,"r") #open ps201 datafile as read only
        # Read header line
        self.header = inpfile.readline().strip()
        # Header line is different between VAX and JAVA versions. 
        # VAX version is all uppercase where JAVA version is mixed case.
        # So test text case to determine which one 
        # JAVAversion true if file was created by JAVA application
        JAVAversion = not self.header.upper() == self.header 

        # Extract data from header
        self.date = self.header.upper().split(" AT ")[1]

        # Extract beam gate from header
        if JAVAversion:
            self.beamgate = self.header.split("(")[0].split("  ")[1]
        else:
            self.beamgate = self.header.split()[1]

        if not (self.beamgate in (lcs.get_pos_beams() + lcs.get_neg_beams())):
            print "Beam gate ", self.beamgate, " not valid, exiting"
            exit()

        self.cavity = self.header.upper().split("CAVITY ")[1].split(" ")[0]
        if not JAVAversion and self.cavity == 'LBPB':
            self.cavity = 'PB'

        # Extract peak beam current from next input line
        self.pkcur = float(
            inpfile.readline().split("=")[1].strip().upper().rstrip("MA")) 

        # Extract beam gate length from next input line
        self.gatelen = int(inpfile.readline().split("=")[1].strip())

        # Extract beam rep rate form next input line
        self.reprate = int(inpfile.readline().split("=")[1].strip())

        # Extract background current from next input line
        if JAVAversion: 
            self.bgcur = float(inpfile.readline().split("=")[1].strip())
        else:
            self.bgcur = 0.0

        # Extract actuator status from next input line
        if JAVAversion:
            self.actuator = map(str.strip,inpfile.readline().split(",")) 
        else:
            self.actuator = ["Not Provided","Not Provided"]

        # Extract PSP target center in later JAVA version
        inpline = inpfile.readline() 
        if JAVAversion:
            if inpline.find('center') > -1:
                self.target = float(inpline.split("=")[1].strip())
                inpline = inpfile.readline()
            else:
                self.target = ""

        # Extract Cavities and RF status from next input line 
        cavs, timing = zip(*map(lambda x: x.split("-"), inpline.split()[1:]))
        # Dictionary keys are the cavities, i.e. PB, MB, T1, T2, T3 and T4
        self.cavs = cavs
        # RF On(0)/Off(1) dictionary
        self.rf = dict(zip(cavs, timing))
        # Make RF status consistent with EPICS channels
        # PS201 uses T for "In-Iime" or "ON" (0) and D for "Delayed" or "OFF" (1)
        for key in cavs: 
            if self.rf[key] == 'T':
                self.rf[key] = lcs.rf_on_val(key)
            else:
                self.rf[key] = lcs.rf_off_val(key)

        # Extract Cavity amplitude set points (ADS counts or %) from next input line
        ampline = map(float, inpfile.readline().split()[1:])            
        # Convert VAX amp values from ADS cnts to % (same as JAVA)
        if JAVAversion:
            self.asp = dict(zip(cavs,ampline)) #Amp Setpoint dictionary
        else:
            if self.beamgate in lcs.get_neg_beams():
                ampsclfac = [0.001, 0.001] + 4*[0.05]
            elif self.beamgate in lcs.get_pos_beams():
                ampsclfac = [0.05, 0.001] + 4*[0.05]
            else:
                print "Beamgate ", 
                self.beamgate, 
                " is not LB or IP/LA, exiting (please update PS201 python class)"
                exit()

            # Convert amplitude values to % of full scale
            self.asp = dict(zip(cavs,map(lambda x, y: x*y, ampline, ampsclfac)))

        # Extract PHASE ADS counts line from VAX file only
        if not JAVAversion:
            inpfile.readline() 
        # Extract Cavity Phase set points (deg) from next input line
        phsline = map(float, inpfile.readline().split()[2:])
        # Phs setpoint dictionary
        self.psp = dict(zip(cavs,phsline))
        # set self.target if ==""
        if self.target == "":
            self.target = self.psp[self.cavity]

        # List of lists (Cav, Timing, Amp, Phs)
        self.cavsp = zip(cavs, timing, ampline, phsline) 

        # Start and Stop beamline elements for each phase scan, i.e. PB to
        # intermediate point then to ABS/COL. Interate from intermediate point
        # to save calculation time
        if self.beamgate in lcs.get_pos_beams():
            self.hps_start = dict(zip(cavs, ['TADB01']*6)) 
            self.hps_intermediate = dict(zip(cavs, [None]*3 + \
                                                 ['02WS01', '03WS01', '04WS01']))
        elif self.beamgate in lcs.get_neg_beams():
            self.hps_start = dict(zip(cavs, ['TBDB02']*6))
            self.hps_intermediate = dict(zip(cavs, [None]*3 + \
                                                 ['02WS01', '03WS01', '04WS01']))
        self.hps_stop = dict(zip(cavs, ['03AB02']*4 + ['04AB01', 'TRAB01']))

        # Absorber energy cut for beam analysis - see protonRangeinCopper.xlsx 
        self.wcut = dict(zip(cavs, [0.0]*3 + [35.86, 65.07, 95.25]))

        #dictionaries of EPICS channels by cavity
        self.epics_pv_asp = dict(
            zip(cavs, [lcs.get_pv_asp(x, self.beamgate) for x in cavs]))
        self.epics_pv_psp = dict(
            zip(cavs, [lcs.get_pv_psp(x, self.beamgate) for x in cavs]))
        self.epics_pv_rf = dict(
            zip(cavs, [lcs.get_pv_rf(x, self.beamgate) for x in cavs]))

        #dictionary of model db fields by cavity - must be consistent with db element names
        if self.beamgate in lcs.get_pos_beams():
            self.model_db_dev = dict(
            zip(cavs, ['TADB01', 'TDDB01', '01LN', '02LN', '03LN', '04LN']))
        elif self.beamgate in lcs.get_neg_beams():
            self.model_db_dev = dict(
            zip(cavs, ['TBDB02', 'TDDB01', '01LN', '02LN', '03LN', '04LN']))
            
        self.model_db_phsoff = dict(zip(cavs, ['phase_offset_cal']*6))
        self.model_db_ampsf = dict(zip(cavs, [""]*2 + ["amplitude_scale_cal"]*4))

#
# create list "fit_parms" containing model_db fitting parameter record names, field names  and values for phasescan
# self.cavity is the shorthand name of the device
#
        if self.cavity == 'PB': 
            # PB phsoff only
            self.fit_parms = [[self.model_db_dev[self.cavity], 
                               self.model_db_phsoff[self.cavity], 
                               ""]]
        elif self.cavity == 'MB':
            # MB phsoff & T1 ampsf only
            self.fit_parms = [[self.model_db_dev['MB'], 
                               self.model_db_phsoff['MB'],
                               ""],
                              [self.model_db_dev['T1'], 
                               self.model_db_ampsf['T1'], 
                               ""]]
        elif self.cavity == 'T1':
            print self.rf['PB'], lcs.rf_off_val('PB')
            if self.rf['PB'] == lcs.rf_off_val('PB'): 
                # MB phsoff & T1 ampsf only
                self.fit_parms = [[self.model_db_dev['MB'], 
                                   self.model_db_phsoff['MB'],
                                   ""],
                                  [self.model_db_dev[self.cavity], 
                                   self.model_db_ampsf[self.cavity], 
                                   ""]]
            else: 
                # PB phsoff, MB phsoff, T1 ampsf
                self.fit_parms = [[self.model_db_dev['PB'], 
                                   self.model_db_phsoff['PB'], 
                                   ""],
                                  [self.model_db_dev['MB'], 
                                   self.model_db_phsoff['MB'],
                                   ""],
                                  [self.model_db_dev[self.cavity], 
                                   self.model_db_ampsf[self.cavity], 
                                   ""]]
        else: 
            # Tank 2-4 phsoff & ampsf
            self.fit_parms = [[self.model_db_dev[self.cavity], 
                               self.model_db_phsoff[self.cavity],
                               ""],
                              [self.model_db_dev[self.cavity], 
                               self.model_db_ampsf[self.cavity], 
                               ""]]

        # Read header lines before actual phase scan data
        inpline = inpfile.readline()
        self.npts = int(inpfile.readline().split("=")[1])
        inpline = inpfile.readline() # Phase ...
        inpline = inpfile.readline() # === ...
        # Read phase scan data
        inpline = inpfile.readline().split()
        temp = []
        while len(inpline) > 1:
            # (phase, current) where current appears to always be positive
            temp.append(map(float, inpline)) 
            inpline = inpfile.readline().split()
        #
        self.data = np.array(temp, dtype=np.float)
        inpfile.close()
#
# End of init_from_file
#
    def init_for_scan(self, cavity_to_scan = 'T1', PB_delayed = True, scan_with_beam = None):
        """Define various quantities need to perform a phase scan on existing 
        dtl parameters. This does not require a .ps201 input file and is done
        in place of that simulation
        tank to scan = 'PB', 'MB', 'T1', 'T2', 'T3', or 'T4'
        PB_delayed = True or False
        scan_with_beam = 'H+', 'IP', 'H-', 'LB'
        default is T1 scan with pB delayed"""
        self.header = 'PS201 simulation with current set points'
        self.date = datetime.today().isoformat()
        self.beamgate = scan_with_beam
        if not (self.beamgate in (lcs.get_pos_beams() + lcs.get_neg_beams())):
            print "Beam gate ", self.beamgate, " not valid, exiting"
            exit()
        cavs = ['PB', 'MB', 'T1', 'T2', 'T3', 'T4']
        self.cavs = cavs
        #dictionaries of EPICS channels by cavity
        self.epics_pv_asp = dict(
            zip(cavs, [lcs.get_pv_asp(x, self.beamgate) for x in cavs]))
        self.epics_pv_psp = dict(
            zip(cavs, [lcs.get_pv_psp(x, self.beamgate) for x in cavs]))
        self.epics_pv_rf = dict(
            zip(cavs, [lcs.get_pv_rf(x, self.beamgate) for x in cavs]))

        #RF in-time/delayed for scans
        timing = ['D', 'T', 'T', 'D', 'D', 'D']
        if not PB_delayed: 
            timing[0] = 'T'
        self.cavity = cavity_to_scan.upper()
        if self.cavity == 'T2':
            timing[3] = 'T'
        elif self.cavity == 'T3':
            timing[3:4] = 2*['T']
        elif self.cavity == 'T4':
            timing[3:5] = 3*['T']
        # RF On(0)/Off(1) dictionary
        self.rf = dict(zip(cavs, timing))
        # Make RF status consistent with EPICS channels
        # PS201 uses T for "In-Iime" or "ON" (0) and D for "Delayed" or "OFF" (1)
        for key in cavs: 
            if self.rf[key] == 'T':
                self.rf[key] = lcs.rf_on_val(key)
            else:
                self.rf[key] = lcs.rf_off_val(key)

        ampline = 6*[0.0]
        phsline = 6*[0.0]
        # store values of all devices from dB in local object
        for i, key in enumerate(cavs):
            ampline[i] = hps.get_db_epics(self.epics_pv_asp[key])
            phsline[i] = hps.get_db_epics(self.epics_pv_psp[key])

        self.asp = dict(zip(cavs, ampline))
        self.psp = dict(zip(cavs, phsline))

        # Start and Stop beamline elements for each phase scan, i.e. PB to
        # intermediate point then to ABS/COL. Interate from intermediate point
        # to save calculation time
        if self.beamgate in lcs.get_pos_beams():
            self.hps_start = dict(zip(cavs, ['TADB01']*6)) 
            self.hps_intermediate = dict(zip(cavs, [None]*3 + \
                                                 ['02WS01', '03WS01', '04WS01']))
        elif self.beamgate in lcs.get_neg_beams():
            self.hps_start = dict(zip(cavs, ['TBDB02']*6))
            self.hps_intermediate = dict(zip(cavs, [None]*3 + \
                                                 ['02WS01', '03WS01', '04WS01']))
        self.hps_stop = dict(zip(cavs, ['03AB02']*4 + ['04AB01', 'TRAB01']))

        # Absorber energy cut for beam analysis - see protonRangeinCopper.xlsx 
        self.wcut = dict(zip(cavs, [0.0]*3 + [35.86, 65.07, 95.25]))

        #dictionary of model db fields by cavity - must be consistent with db element names
        if self.beamgate in lcs.get_pos_beams():
            self.model_db_dev = dict(
            zip(cavs, ['TADB01', 'TDDB01', '01LN', '02LN', '03LN', '04LN']))
        elif self.beamgate in lcs.get_neg_beams():
            self.model_db_dev = dict(
            zip(cavs, ['TBDB02', 'TDDB01', '01LN', '02LN', '03LN', '04LN']))
            
        self.model_db_phsoff = dict(zip(cavs, ['phase_offset_cal']*6))
        self.model_db_ampsf = dict(zip(cavs, [""]*2 + ["amplitude_scale_cal"]*4))
        self.npts = 0

    def set_threshold(self, thres):
        """Set threshold for returning getting data"""
        self.threshold = thres

    def set_no_threshold(self):
        """Set threshold to 0"""
        self.threshold = 0

    def set_phase_limits(self, phimin, phimax):
        """Set min, max phase limits for getting data"""
        self.plimits = [phimin, phimax]

    def set_no_phase_limits(self):
        """Remove phase limits"""
        self.plimits = []

    def get_fit_sf(self):
        """Returns factor used to scale HPSim beam intensity results"""
        return self.fit_sf

    def set_fit_sf(self, val):
        """Set factor used to scale HPSim beam intensity results"""
        self.fit_sf=val

    def init_fit_parms(self):
        """Get fit parameters values from model_db and store in local list"""
        for i,elem in enumerate(self.fit_parms):
            name, field, val = elem
            val = hps.get_db_model(name, field)
            self.fit_parms[i][2] = val

    def print_fit_parms(self):
        """Print fit parameter list"""
        print self.cavity," phase scan fit parameters"
        for elem in self.fit_parms:
            print elem

    def title_fit_parms(self):
        """Create Title string containing complete fit parameter list"""
        title = "Scale Factor=" + str(self.get_fit_sf()) + '\n'
        print self.cavity," phase scan fit parameters"
        for i in self.get_fit_parms():
            title = title + i[0] + ' '+ i[1] + '=' + str(i[2]) +';\n'
        return title[:-2]

    def get_fit_parms(self):
        """Return list of fitting parameter names,fields and values"""
        print "get_fit_parms"
        return self.fit_parms

    def get_fit_parm_vals(self):
        """Return list of bare parameter values from fit_parms list"""
        print "get_fit_parm_vals"
        return list(zip(*self.fit_parms)[2])

    def set_fit_parm_vals(self, pval):
        """Store list of bare parameter values in fit_parms list"""
        for i, elem in enumerate(self.fit_parms):
            self.fit_parms[i][2]=pval[i]

    def store_fit_parms(self, parms):
        """Update model_db fit parameter with user supplied values"""
        for elem in parms:
            name, field, val = elem
            print name, field, val
            hps.set_db_model(name, field, str(val))

    def update_fit_parms(self):
        """Update model_db with latest fit parmeter values"""
        self.store_fit_parms(self.fit_parms)

    def init_all_cavs(self):
        """Apply EPICS set points from PS201 file to all 201 MHz cavities"""
        for key in self.cavs:
            # RF on(0)/off(1)
            hps.set_db_epics(self.epics_pv_rf[key], self.rf[key])
            # Phase set point
            hps.set_db_epics(self.epics_pv_psp[key], str(self.psp[key]))
            # Amp set point
            hps.set_db_epics(self.epics_pv_asp[key], str(self.asp[key])) 

    def init_all_cavs_for_ps_sim(self):
        """Use Apply EPICS set points from PS201 file to all 201 MHz cavities"""
        for key in self.cavs:
            # RF on(0)/off(1)
            hps.set_db_epics(self.epics_pv_rf[key], self.rf[key])
            # Phase set point
            hps.set_db_epics(self.epics_pv_psp[key], str(self.psp[key]))
            # Amp set point
            hps.set_db_epics(self.epics_pv_asp[key], str(self.asp[key])) 

    def set_pb_phase_only_fit(self):
        """This changes T1 w/PB:ON to only fit PB phase offset"""
        if self.cavity == 'T1': self.fit_parms = [self.fit_parms[0]] 

    def get_all_cavs(self):
        """Returns from db a list of EPICS set points for all 201 MHz cavities"""
        L = []
        for key in self.cavs:
            # this first line doesn't work, needs change in sql db
            # RF on/off
            rf = hps.get_db_epics(self.epics_pv_rf[key])
            # Phase setpoint
            psp = hps.get_db_epics(self.epics_pv_psp[key]) 
            # Amp setpoint
            asp = hps.get_db_epics(self.epics_pv_asp[key])
            L.append([key, ['RF', self.epics_pv_rf[key], rf], 
                          ['PSP', self.epics_pv_psp[key], psp], 
                          ['ASP', self.epics_pv_asp[key], asp]])
        return L

    def set_all_cavs(self, splist):
        """Set all 201 MHz cavities db PV's to valuesin splist"""
        for item in splist:
            for elem in item[1:]:
                pv = elem[1]
                val = elem[2]
                hps.set_db_epics(pv, val)

    def optimize_ps_fit(self, beam, sim, xdata, ydata):
        """Returns fitting parameters (data_sf, phase_offset, amp_sf) 
        that produced best fit in least-squares sense using Scipy Optimize"""
        key = self.cavity
        parms = [self.get_fit_sf()] + self.get_fit_parm_vals()
        print '*** Starting PS201 Fitting Procedure ***'
        if self.hps_intermediate[key] is None:
            print " from ", self.hps_start[key], " to ", self.hps_stop[key]
        else:
            print " from ", self.hps_start[key]," to ", self.hps_intermediate[key]
            print " then interate from", hps.get_next_element(self.hps_intermediate[key]), "to ", self.hps_stop[key]

        def fit_func(parms, x):
            print "Current Fit parameters: ", parms
            fit_sf=parms[0]
            fit_parms=parms[1:]
            # Scale factor between fit and data
            self.set_fit_sf(fit_sf) 
            # Update values in local list
            self.set_fit_parm_vals(fit_parms) 
            # Update values in model_db
            self.update_fit_parms() 
            L=[]
            if self.hps_intermediate[key] is not None:
                # Restore input beam
                beam.restore_initial_beam()
                # Simulate from start to intermediate location
                sim.simulate(self.hps_start[key],self.hps_intermediate[key])
                # save intermediate beam
                beam.save_intermediate_beam()

            for phase in x:
                if self.hps_intermediate[key] is None:
                    # Start from beginning so...
                    # Restore input beam
                    beam.restore_initial_beam()
                    # Set phasescan cavity phase
                    hps.set_db_epics(self.epics_pv_psp[key], str(phase))
                    # Simulate from start to stop
                    sim.simulate(self.hps_start[key], self.hps_stop[key])
                else:
                    # Restore intermediate beam
                    beam.restore_intermediate_beam()
                    # Set phasescan cavity phase
                    hps.set_db_epics(self.epics_pv_psp[key], str(phase))
                    # Simulate from next element after intermediate location to end
                    sim.simulate(hps.get_next_element(self.hps_intermediate[key]), self.hps_stop[key])

                # Get current of beam particles whose energies are above ABS/COLL threshold
                beam_current = self.get_fit_sf() * self.get_beam_above_ethreshold(beam)
                #beam_current = self.get_fit_sf() * beam.get_current()
                L.append(beam_current)
            return np.array(L, dtype=np.float)

        def err_func(parms, x, y):
            # x, y inputs are Numpy arrays
            ysim = fit_func(parms, x)
#            err = map(lambda (x): x[0]-x[1], zip(ysim, y))
            err = y - ysim
#           print "err^2", sum(map(lambda (x): x*x, err))
            print "err^2: ", np.sum(err*err), '\n'
            return err

        pguess = parms
        pfinal, success = optimize.leastsq(err_func, pguess, 
                                           args=(xdata, ydata), epsfcn=0.001)
        print 'fit success', success
        return pfinal[1:]

    def simulate_ps(self, beam, sim, xdata):
        """Returns the simulated PS201 beam current values data"""
        key = self.cavity
        yresult = []
        print "*** Starting PS201 Simulation ***"
        if self.hps_intermediate[key] is None:
            print " from ", self.hps_start[key], " to ", self.hps_stop[key]
        else:
            print " from ", self.hps_start[key]," to ", self.hps_intermediate[key]
            print " then interate from", hps.get_next_element(self.hps_intermediate[key]), "to ", self.hps_stop[key]
        if self.hps_intermediate[key] is not None:
            # Restore input beam
            beam.restore_initial_beam()
            # Simulate from start to intermediate location
            sim.simulate(self.hps_start[key],self.hps_intermediate[key])
            # save intermediate beam
            beam.save_intermediate_beam()

        for phase in xdata:
            if self.hps_intermediate[key] is None:
                # Restore input beam
                beam.restore_initial_beam()
                # Set phasescan cavity phase
                hps.set_db_epics(self.epics_pv_psp[key],str(phase))
                # Simulate from start to intermediate location
                sim.simulate(self.hps_start[key],self.hps_stop[key])
            else:
                # Restore intermediate beam
                beam.restore_intermediate_beam()
                # Set phasescan cavity phase
                hps.set_db_epics(self.epics_pv_psp[key],str(phase))
                # Simulate from next element after intermediate location to end
                sim.simulate(hps.get_next_element(self.hps_intermediate[key]), self.hps_stop[key])

            # Get current of beam particles whose energies are above Abs/Coll Threshold
            beam_current=self.get_fit_sf()*self.get_beam_above_ethreshold(beam)
            yresult.append(beam_current)
        return np.array(yresult, dtype=np.float)

    def get_beam_above_ethreshold(self, beam):
        """Get current of 'good' beam with energy above abs/coll energy 
        threshold."""
# Assumes beam particle array is list
#        bc = beam.get_current()
#        # Get energy threshold for Abs/Coll used by cavity in measurement
#        ecut = self.wcut[self.cavity]
#        # Get only good particles in output beam
#        elist = beam.get_w("good")
#        total = len(elist)
#        good = sum(map(lambda (x): x > ecut and 1 or 0, elist))
#        if total > 0:
#            return beam.get_current() * good / total
#        else:
#            return 0
        wmask = beam.get_mask_with_limits('w', lolim = self.wcut[self.cavity])
        mask = beam.get_good_mask(wmask)
        return beam.get_current(mask)

    def display_info(self):
        """Print PS201 data file information, exlcuding scan data"""
        print "---PS201 information---"
        print "Header: ", self.header
        print "Date: ", self.date
        print "Cavity: ", self.cavity
        print "Beam Gate: ",self.beamgate
        print "Beam Gate Length: ",self.gatelen
        print "Rep Rate: ",self.reprate
        print "Peak Current: ",self.pkcur
        print "BG Current: ",self.bgcur
        print "Actuator: ",self.actuator
        print "Cavity Settings \n (ID, Time, Amp, Phs): "
        for cav in self.cavsp: print cav
        print "Number points: ", self.npts

    def display_data(self):
        """Print PS201 scan data"""
        print "---Original PS201 data---"
        print "Phase, Current"
        for phi, curr in self.data:
            print phi, curr

    def display_fit_info(self):
        """Print PS201 fit info"""
        print "---PS201 fit information---"
        print "Threshold= {0:.1f}" .format(self.threshold)
        print "Window [pmin, pmax] [deg]=", self.plimits
        print "Abs/Coll Energy Threshold [MeV]=", self.wcut[self.cavity]
        print "Fit scale factor=", self.fit_sf
        print "Fit parameters"
        for item in self.fit_parms:
            print item
        print "EPICS PV's"
        print "Cav   ASP        PSP        RF"
        for key in self.cavs:
            print key, self.epics_pv_asp[key], self.epics_pv_psp[key], \
                self.epics_pv_rf[key]

    def get_data(self):
        """Returns xy-pair Numpy array of scan data, with (0,0) pairs removed,
        at or above threshold and within user define phaselimits"""
        return npu.apply_limits(npu.apply_threshold(
                npu.remove_zero_pairs(self.data), self.threshold), self.plimits)
              
    def get_phase_data(self):
        """Returns a Numpy array containing phase values of measured points
        after cuts applied"""
        return self.get_data()[:, _X]

    def get_curr_data(self):
        """Returns a Numpy array containing intensity values of measured points 
        after cuts applied"""
        return self.get_data()[:, _Y]

#23456789c123456789c123456789c123456789c123456789c123456789c123456789c123456789c
    def get_halfmax_phs(self):
        """Returns a pair of phase values corresponding to the half-maximum points"""
        xy = self.get_data()
        phslo = npu.get_halfmax_x(xy, 'lower')
        phsup = npu.get_halfmax_x(xy, 'upper')
        return [phslo, phsup]

    def get_fwhm(self):
        """Returns FWHM (deg) of phase scan distribution"""
        phslo,phsup=self.get_halfmax_phs()
        return phsup-phslo

    def get_dcle(self):
        """Returns the Dist from Center to Left Edge of phase scan"""
        phslo,phsup=self.get_halfmax_phs()
        if self.target <> "":
            phscent=self.target
        #else:
            phscent=self.psp[self.cavity]
#        print phslo, phscent, phsup
        return phscent - phslo

    def display_results(self):
        """Show results of PS201 analysis"""
        print "---Analysis Results for PS201 scan---"
        print self.header
        print "FWHM = {0:.2f} deg".format(self.get_fwhm())
        print "DCLE= {0:.2f} deg" .format(self.get_dcle())
        if self.cavity in ['PB', 'MB', 'T1']:
            print "PVR= {0:.2f}" .format(npu.get_pvr(self.get_data()))
        print "Threshold= {0:.1f}" .format(self.threshold)
        print "Window [pmin, pmax]=", self.plimits

    def get_results(self):
        """Return a string containing the results of PS201 analysis"""
        output =  "---Analysis Results for PS201 scan---" + '\n'
        output += "FWHM = {0:.2f} deg".format(self.get_fwhm()) + '\n'
        output += "DCLE = {0:.2f} deg" .format(self.get_dcle()) + '\n'
        if self.cavity in ['PB', 'MB', 'T1']:
            output += "PVR = {0:.2f}".format(npu.get_pvr(self.get_data())) + '\n'
        output += "Threshold= {0:.1f}" .format(self.threshold) + '\n'
        output += "Window [pmin, pmax] = " + str(self.plimits) + '\n'
        return output

### end of PS201 class and methods
