#hpsim.py
# 20160526 - ljr - v7, add 'bin' path to get correct version HPSim
# 20160303 - ljr - v6, updated to work with latest version of HPSim
# 20160222 - ljr - v5, bug fixes, code cleanup
# 20160222 - ljr - v4, bug fixes, code cleanup
# 20160217 - ljr - v3, bug fixes, code cleanup
# 20160211 - ljr - v2, bug fixes
# 20160202 - ljr - updated get_current
"""Python wrapper for HPSim.so funtions. This version utilizes Numpy arrays 
and adds some functionality. The DBConnection, Beamline and SpaceCharge objects 
are hidden in the classes and used implicitly when calling functions that 
depend upon them.
"""
import sys
import os
# add import paths 
par_dir = os.path.abspath(os.path.pardir)
sys.path.append(par_dir)
lib_dir = os.path.join(par_dir,"bin") #HPSim.so
sys.path.append(lib_dir)
pkg_dir = os.path.join(par_dir,"pylib") #HPSim related packages
sys.path.append(pkg_dir)

#import various packages
import HPSim as HPSim
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter, Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import lcsutil_v2 as lcs

import matplotlib as mpl

#global variables
_COORDINATES = ['x', 'xp', 'y', 'yp', 'phi', 'w']
_EMITTANCE = ['x', 'y', 'z']
_PHASESPACE = ['xxp', 'yyp', 'phiw']
_LOSSES = ['losses']
# units conversion factors
_CM = 100.0
_MR = 1000.0
_DEG = 180.0/math.pi
_RAD = 1/_DEG
_MA = 1000.0 
_USER_UNITS = {'x':_CM, 'xp':_MR, 'y':_CM, 'yp':_MR, 'phi':_DEG, 'w':1.0}
_USER_LABELS = {'x':'cm', 'xp':'mr', 'y':'cm', 'yp':'mr', 'phi':'deg', 'w':'MeV',
                'i':'mA'}

class Beam():
    """An hpsim class for manipulating beams as Numpy arrays"""

# functions from original HPSim.so 

    def __init__(self, **args): #mass, charge, current, file = None, num = 0):
        """Creates an instance of the beam class

        Args:
           keywords are required and lowercase
           where
           
           file (string): Input file name containing beam distribution

            or

           mass (double): mc^2 of particle mass (MeV)
           charge (double): q/|e|
           current (double): bunch peak current(Amps)
           num (int): number of macro particles

        Attributes:
           initial_current(double): Initial beam current (A)
           initial_frequency(double): Initial freq (MHz)
           initial_size(int): Initial number of macroparticles in beam distr

        Returns:
            Returns a beam class object

        Examples:
            Beam(file = filename) 
            
             or
            
            Beam(mass = particle_mass, charge = particle_charge, 
                           current = beam_current, num = number of particles)

        """
        if 'file' in args.keys():
            # create beam based upon contents of file
            if os.path.isfile(args['file']):
                self.beam = HPSim.Beam(file = args['file'])
                self.initial_current = self.beam.get_current()
                self.initial_frequency = self.beam.get_frequency()
                self.initial_gd_size = self.beam.get_size() - self.beam.get_loss_num()
            else:
                print 'Input beam file, "',args['file'],'" not found! Exiting'
                exit()
        else:
            # create beam from supplied arguments
            self.beam = HPSim.Beam(mass=float(args['mass']),
                                   charge=float(args['charge']), 
                                   current=float(args['current']),
                                   num=int(args['num']))
            self.initial_current = float(args['current'])
            self.initial_frequency = 0
            self.initial_gd_size = self.get_initial_size()
        return
     
    def get_distribution(self, option='all'):
        """Returns a list of Numpy vectors containing the beam coordinates in 
        user units x, xp, y, yp, phi, w, loss = beam.get_distribution()

        Argument:
           option (str): 'good', 'bad' or 'all'=default
        """
        loss_all = self.get_losses()
        if option == 'all':
            loss = loss_all
        elif option == 'bad':
            loss = np.where(loss_all > 0)[0]
        elif option == 'good':
            loss = np.where(loss_all == 0)[0]
        return self.get_x(option) * _CM, \
            self.get_xp(option) * _MR, \
            self.get_y(option) * _CM, \
            self.get_yp(option) * _MR, \
            self.get_phi(option) * _DEG, \
            self.get_w(option), \
            loss

    def set_distribution(self, x, xp, y, yp, phi, w, loss = None):
        """Creates beam distribution using vectors of coordinates (users units)
        Arguments:
           x (Numpy vector double): x coordinates cm
           xp (Numpy vector double): xp coordinates mr
           y (Numpy vector double): y coordinates cm
           yp (Numpy vector double): yp coordinates mr
           phi (Numpy vector double): phi coordinates deg
           w (Numpy vector double): w coordinates MeV
           loss (Numpy vector int, optional): loss coordinate or 0-> good 
        """
        # first zero out distribution if array will be only partially filled
        if len(x) < self.beam.get_size():
            zeros = self.beam.get_size()*[0.0]
            self.beam.set_distribution(zeros, zeros, zeros, zeros, zeros, zeros)
        if loss is not None:
            self.beam.set_distribution(list(x/_CM), list(xp/_MR), 
                                       list(y/_CM), list(yp/_MR),
                                       list(phi/_DEG), list(w), list(loss))
        else:
            self.beam.set_distribution(list(x/_CM), list(xp/_MR), 
                                       list(y/_CM), list(yp/_MR),
                                       list(phi/_DEG), list(w))            
        return
    
    def set_waterbag(self, alpha_x, beta_x, emittance_x,
                     alpha_y, beta_y, emittance_y,
                     alpha_z, beta_z, emittance_z,
                     sync_phi, sync_w, frequency, random_seed = 0):
        """ Creates a 6D waterbag using PARMILA input units
        Arguments:
           alpha_x (double): x-plane Twiss alpha parameter
           beta_x (double): x-plane Twiss beta parameter (cm/radian)
           emittance_x (double): x-plane total emittance (cm * radian)
           alpha_y (double): y-plane Twiss alpha parameter
           beta_y (double): y-plane Twiss beta parameter (cm/radian)
           emittance_y (double): y-plane total emittance (cm * radian)
           alpha_z (double): z-plane Twiss alpha parameter
           beta_z (double): z-plane Twiss beta parameter (deg/MeV)
           emittance_z (double): z-plane total emittance (deg * MeV)
           synch_phi (double): synchronous phase (deg)
           synch_w (double): synchronous energy (MeV)
           frequency (double): frequency (MHz)
           random_seed (option [int]): random seed for generating distribution
        """
        if random_seed:
            self.beam.set_waterbag(float(alpha_x), float(beta_x), float(emittance_x),
                             float(alpha_y), float(beta_y), float(emittance_y),
                             float(alpha_z), float(beta_z*_DEG), float(emittance_z/_DEG),
                             float(synch_phi/_DEG), float(synch_w), float(frequency),
                             int(random_seed))
        else:
            self.beam.set_waterbag(float(alpha_x), float(beta_x), float(emittance_x),
                             float(alpha_y), float(beta_y), float(emittance_y),
                             float(alpha_z), float(beta_z*_DEG), float(emittance_z/_DEG),
                             float(synch_phi/_DEG), float(synch_w), float(frequency))
        return
    
    def set_dc(self, alpha_x, beta_x, emittance_x,
               alpha_y, beta_y, emittance_y,
               delta_phi, synch_phi, synch_w, random_seed = 0):
        """ Creates DC beam using PARMILA input units set_waterbag
        Arguments:
           alpha_x (double): x-plane Twiss alpha parameter
           beta_x (double): x-plane Twiss beta parameter (cm/radian)
           emittance_x (double): x-plane total emittance (cm * radian)
           alpha_y (double): y-plane Twiss alpha parameter
           beta_y (double): y-plane Twiss beta parameter (cm/radian)
           emittance_y (double): y-plane total emittance (cm * radian)
           alpha_z (double): z-plane Twiss alpha parameter
           beta_z (double): z-plane Twiss beta parameter (deg/MeV)
           emittance_z (double): z-plane total emittance (deg * MeV)
           delta_phi (double): half-width of phase distribution (deg)
           synch_phi (double): synchronous phase (deg)
           synch_w (double): synchronous energy (MeV)
           random_seed (int, optional): random seed for generating distribution
        """
        if random_seed:
            self.beam.set_dc(float(alpha_x), float(beta_x), float(emittance_x),
                       float(alpha_y), float(beta_y), float(emittance_y),
                       float(delta_phi/_DEG), float(synch_phi/_DEG), float(synch_w),
                       int(random_seed))
        else:
            self.beam.set_dc(float(alpha_x), float(beta_x), float(emittance_x),
                       float(alpha_y), float(beta_y), float(emittance_y),
                       float(delta_phi/_DEG), float(synch_phi/_DEG), float(synch_w))
        return
    
    def save_initial_beam(self):
        """Save initial beam distribution for later restore."""
        self.beam.save_initial_beam()
        return

    def save_intermediate_beam(self):
        """Save intermediate beam distribution for later restore."""
        self.beam.save_intermediate_beam()
        return
    
    def restore_initial_beam(self):
        """Restore initial beam distribution for next simulation."""
        self.beam.restore_initial_beam()
        return
    
    def restore_intermediate_beam(self):
        """Restore intermediate beam distribution for next simulation."""
        self.beam.restore_intermediate_beam()
        return
    
    def print_simple(self):
        """Print particle coordinates x, x', y, y', phi, w coordinates to screen"""
        self.beam.print_simple()
        return
    
    def print_to(self, output_file_name):
        """Print particle coordinates x, x', y, y', phi, w coordinates to file"""
        self.beam.print_to(output_file_name)
        return
    
    def set_ref_w(self, w):
        """Set reference particle energy, MeV"""
        self.beam.set_ref_w(float(w))
        return
    
    def get_ref_w(self):
        """Return the reference particle's energy in MeV"""
        return self.beam.get_ref_w()

    def set_ref_phi(self, phi):
        """Set reference particle phase, degrees"""
        self.beam.set_ref_phi(float(phi / _DEG))
        return
    
    def get_ref_phi(self):
        """Return the reference particle's phase in degree"""
        return self.beam.get_ref_phi() * _DEG

    def set_frequency(self, frequency):
        """Set beam frequency in MHz"""
        self.beam.set_frequency(float(frequency))
        if self.initial_frequency == 0:
            self.initial_frequency = self.get_frequency()
        return
    
    def get_frequency(self):
        """Return beam frequency in MHz"""
        return self.beam.get_frequency()

    def get_mass(self):
        """Return mass of beam, mc^2, in MeV"""
        return self.beam.get_mass()

    def get_charge(self):
        """Return charge of beam in q/|e|"""
        return self.beam.get_charge()

    def get_initial_size(self):
        """Returns initial number of beam macro particles"""
        return self.beam.get_size()

    def get_x(self, option = 'all'):
        """Return Numpy array of x coordinates (cm) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_x(option) * _CM

    def get_xp(self, option = 'all'):
        """Return Numpy array xp coordinates (mr) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_xp(option) * _MR

    def get_y(self, option = 'all'):
        """Return Numpy array y coordinates of (cm) macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_y(option) * _CM

    def get_yp(self, option = 'all'):
        """Return Numpy array yp coordinates of (mr) macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_yp(option) * _MR

    def get_phi(self, option = 'all'):
        """Return Numpy array phi coordinates (deg) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_phi(option) * _DEG

    def get_w(self, option = 'all'):
        """Return Numpy array w coordinates (MeV) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_w(option)

    def get_losses(self):
        """Return Numpy array of loss condition of macro particles"""
        return self.beam.get_losses()

    def get_loss_num(self):
        """Return number of lost particles"""
        return self.beam.get_loss_num()

    def get_avg_x(self):
        """Return average x value of beam in cm"""
        return self.beam.get_avg_x() * _CM

    def get_avg_y(self):
        """Return average y value of beam in cm"""
        return self.beam.get_avg_y() * _CM

    def get_avg_phi(self, option = 'absolute'):
        """Return average phi value of beam in deg"""
        return self.beam.get_avg_phi(option) * _DEG

    def get_avg_w(self):
        """Return average w value of beam in MeV"""
        return self.beam.get_avg_w()

    def get_sig_x(self):
        """Return sigma x of beam in cm"""
        return self.beam.get_sig_x() * _CM

    def get_sig_y(self):
        """Return sigma y of beam in cm"""
        return self.beam.get_sig_y() * _CM

    def get_sig_phi(self):
        """Return sigma phi of beam in deg"""
        return self.beam.get_sig_phi() * _DEG

    def get_sig_w(self):
        """Return sigma w of beam"""
        return self.beam.get_sig_w()

    def get_emittance_x(self):
        """Return rms x emittance of beam cm*mr"""
        return self.beam.get_emittance_x() * _CM * _MR

    def get_emittance_y(self):
        """Return rms y emittance of beam in cm*mr"""
        return self.beam.get_emittance_y() * _CM * _MR

    def get_emittance_z(self):
        """Return rms z emittance of beam in Deg*MeV"""
        return self.beam.get_emittance_z() * _DEG

#    def get_current(self):
#        """Return beam current, mA"""
#        return self.beam.get_current() * _MA

    def apply_cut(self, axis, minval, maxval):
        """Remove particles from beam by apply cuts along 'x', 'y', 'p' or 'w'"""
        self.beam.apply_cut(axis, minval, maxval)
        return

    def translate(self, axis, value):
        """Translate particle coordinates along specified axis by given value"""
        # divide by to convert from user_units
        self.beam.translate(axis, value / _USER_UNITS[axis.lower()])
        return
    
################################ new functions #################################
# These functions use create or employ masks that allow the user to be 
# selective in what particles are returned or analyzed based upon the 
# mask constraints

    def get_coor(self, var, mask = None):
        """Return vector of macro particle coordinates (in USER_UNITS) 
        after optional mask is applied.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        lc_var = var.lower()
        request = "self.beam.get_" + lc_var
        if lc_var in _COORDINATES:
            request += "('all')"
            if mask is not None:
                coor = eval(request)[mask] * _USER_UNITS[lc_var]
            else:
                coor = eval(request) * _USER_UNITS[lc_var]
            if lc_var == 'phi':
                 coor=coor   
#                coor = modulo_phase(coor, self.get_ref_phi())
        elif lc_var in _LOSSES:
            request += "()"            
            if mask is not None:
                coor = eval(request)[mask]
            else:
                coor = eval(request)
        else:
            print "Error: Empty array returned, variable", \
            str.upper(var), "not recognized"
            return np.array([])
        return coor

    def get_avg(self, var, mask = None):
        """Return average of beam coordinates after optional mask applied.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        lc_var = var.lower()
        if lc_var in _COORDINATES:
            # if mask is an empty list then return average of 0.0
            if mask == None:
                return np.mean(self.get_coor(lc_var, mask))
            elif list(mask) == []:
                return 0.0
            else:
                return np.mean(self.get_coor(lc_var, mask))
        else:
            print "Error: Average not found. Variable", \
            str.upper(var), "not recognized"
            return float('nan')

    def get_sig(self, var, mask = None):
        """Return sigma of beam coordinates after optional mask applied.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        lc_var = var.lower()
        if var.lower() in _COORDINATES:
            return np.std(self.get_coor(lc_var, mask))
        else:
            print "Error: Sigma not found. Variable", \
            str.upper(var), "not recognized"
            return float('nan')
    
    def get_twiss(self, var, mask = None):
        """Return Twiss parameters (a,b, unnormalized, rms e) for specified coords x, y or z.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """

        lc_var = var.lower()
        if lc_var in _EMITTANCE:
            v = lc_var
            vp = v + 'p'
            if v == 'z':
                v = 'phi'
                vp = 'w'
            avgv2 = math.pow(self.get_sig(v, mask), 2)
            avgvp2 = math.pow(self.get_sig(vp, mask), 2)
            avgvvp = np.mean((self.get_coor(v, mask) - self.get_avg(v, mask)) *
                               (self.get_coor(vp, mask) - self.get_avg(vp, mask)))
            ermssq = avgv2 * avgvp2 - avgvvp * avgvvp
            if ermssq > 0:
                erms = math.sqrt(ermssq)
                alpha = - avgvvp / erms
                beta = avgv2 / erms
                return [alpha, beta, erms]            
            else:
                #print "Error: " + str.upper(var) + " emittance undefined"
                return 3*[float('NaN')]

    def get_urms_emit(self, var, mask = None):
        """Return unnormalized rms emittance along specified axis, x, y or z.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        a, b, e_urms = self.get_twiss(var, mask)
        return e_urms

    def get_nrms_emit(self, var, mask = None):
        """Return normalized rms emittance along specified axis, x, y or z.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        if var in ['x', 'y']:
            return self.get_urms_emit(var, mask) * self.get_betagamma(mask)
        else:
            return self.get_urms_emit(var, mask)

    def get_mask_with_limits(self, var, lolim, uplim = None):
        """Creates a a mask, i.e. a Numpy vector of a list of indices, based upon 
        variable x, xp, y, yp, phi, w or losses above lower limit and below 
        optional upper limit. User units

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           lolim (double): lower limit, above which, particles are included in mask
           uplim (double, optional): upper limit, below which, particles are included
           in mask.
        """

        lc_var = var.lower()
        request = "self.get_" + str(lc_var)
        if lc_var in _COORDINATES:
            request += "('all')"
        elif lc_var in _LOSSES:
            request += "()"
        else:
            print "Error: Empty masked returned, variable", str.upper(var), "not recognized"
            return np.array([])

        aray = eval(request)
        # create test string for later eval in np.where function
        test = "(aray > " + str(lolim) + ")"
        if uplim is not None:
            test += " & (aray < " + str(uplim) + ")"

        return np.where(eval(test))[0]

   
    def get_good_mask(self, mask = None):
        """Returns indices of particles not lost.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        # get_losses = 0 if particle is still good, else element number where it was lost
        if mask is not None:
            return np.intersect1d(mask, np.where(self.beam.get_losses() == 0)[0])
        else:
            return np.where(self.beam.get_losses() == 0)[0]

    def get_lost_mask(self, mask = None):
        """Returns indices of particles not lost.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        # get_losses = 0 if particle is still good, else element number where it was lost
        if mask is not None:
            return np.intersect1d(mask, np.where(self.beam.get_losses() != 0)[0])
        else:
            return np.where(self.beam.get_losses() != 0)[0]

    def get_intersection_mask(self, mask1, mask2):
        """Returns the mask that results from the intersection of two masks.
        Arguments:
           mask1 (Numpy vector): mask with condition 1 used to select particles
           mask2 (Numpy vector): mask with condition 2 used to select particles
        """

        return np.intersect1d(mask1, mask2)
        
    def get_betagamma(self, mask = None):
        """Return value of beta*gamma of beam.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        if mask == None:
            gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
            return  math.sqrt(gamma * gamma -1.0)
        elif list(mask) == []:
            return 0.0
        else:
            gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
            return  math.sqrt(gamma * gamma -1.0)

    def get_betalambda(self, mask = None):
        """Return value of beta*lambda of beam.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        c = 2.99792458e8 # m/s
        wavelength = c / (self.get_frequency() * 1.0e6)
        if mask == None:
            gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
            beta = math.sqrt(1.0 - 1/(gamma * gamma))
        elif list(mask) == []:
            beta = 0.0
        else:
            gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
            beta = math.sqrt(1.0 - 1/(gamma * gamma))
        return beta * wavelength

    def get_current(self, mask = None):
        """Return beam current in user units of beam"""
        # the original HPSim get_current returns the remaining beam
        # current associated with the 'good' particles
        # need to scale this result
        if mask is None:
            return self.beam.get_current() * _MA
        else:
            return self.initial_current * _MA *  \
                self.get_frequency() / self.initial_frequency *  \
                self.get_size(mask) / self.initial_gd_size
        
    def get_size(self, mask = None):
        """Return number of beam particles with or w/o mask applied
           Without a mask: Returns number of 'good' particles, i.e. not lost
           With a mask: Returns the length of the mask, i.e. number that satisfy mask.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        if mask == None:
            # return number of good particles, i.e. not lost transversely
            # when mask in not present
            return self.get_initial_size() - self.beam.get_loss_num()
        else:
            #return length of mask - numpy array when mask is present
            return mask.size

    def print_results(self, mask = None):
        """Prints avg, sigma, alpha, beta, Eurms, Enrms for all coord of distr.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        print "Distribution Analysis Results (w/user units)"
        print 'Mass = {0:.4f}'.format(self.get_mass())
        print 'Charge/|e| = {0:.0f}'.format(self.get_charge())
        print 'Ib = {0:.2f} {1}'.format(self.get_current(mask), _USER_LABELS['i'])
        print 'Frequency = {0:.3f} MHz'.format(self.get_frequency())
        if mask is not None:
            print '*** Mask applied ***'
            print 'Number of macroparticles(mask) = {0:.0f}'.format(self.get_size(mask))
            print 'Number of macroparticles lost  = {0:.0f}'.format(self.beam.get_loss_num())
        else:
            print '*** No Mask applied ***'
            print 'Number of macroparticles(good) = {0:.0f}'.format(self.get_size())
            print 'Number of macroparticles lost  = {0:.0f}'.format(self.beam.get_loss_num())

        print 'Ref part. \n phi = {0:10.4f} {1}\n   w = {2:10.4f} {3}'.\
            format(self.get_ref_phi(), _USER_LABELS['phi'],
                   self.get_ref_w(), _USER_LABELS['w'])
        if self.get_size(mask) > 0:
            print '\nCentroids and RMS sizes'
            print '            Avg         Sigma'
            for item in _COORDINATES:
                print '{0:3}: {1:10.4f}    {2:10.4f} {3:3}'\
                    .format(item, self.get_avg(item, mask),
                        self.get_sig(item, mask), _USER_LABELS[item])
            print '\nTwiss parameters'

            print '         Alpha      Beta      Eurms      Enrms'
            for item in _EMITTANCE:
                a, b, eurms = self.get_twiss(item, mask)
                print '{0:2}: {1:10.4f} {2:10.4f} {3:10.4f} {4:11.5f}'\
                    .format(item, a, b, eurms, self.get_nrms_emit(item, mask))
            print '\n'
        else:
            print '\n*** No particles remaining ****'
            print '\n'
        return
    
################################################################################

class Distribution():
    """An hpsim class for holding a masked beam of particles as a static np-array. 
    Faster for analysis and plotting than using beam array
    """

    coor_index = dict(zip(_COORDINATES + _LOSSES, range(0, len(_COORDINATES + _LOSSES))))
    def __init__(self, beam, mask = None):
        """Init creates an instance of the Distribution object containing all 
        the vectors of coordinates from the beam object that satisfy the mask

        Attributes:
           mass (double): mc^2 of particle double mass (MeV)
           charge (double): q/|e|
           current (double): bunch peak current(Amps)
           frequency (double): MHz
           size (int): number of macro particles
           betagamma (double): beta * gamma of masked beam
           betalambda (dounble): beta * lambda of masked beam
           ref_phi (double): reference particle phase (Rad)
           ref_w (double): reference particle energy (MeV)

        Returns:
            Returns a beam distribution object

        """
        # create np float array of appropriate size
        if mask is None:
            bm_size = beam.get_initial_size()
        else:
            bm_size = beam.get_size(mask)

        self.coor = np.zeros([len(_COORDINATES + _LOSSES), bm_size])
        for coor in _COORDINATES + _LOSSES:
            ndx = Distribution.coor_index[coor]
            self.coor[ndx] = beam.get_coor(coor, mask=mask)

        self.current = beam.get_current(mask)
        self.frequency = beam.get_frequency()
        self.mass = beam.get_mass()
        self.charge = beam.get_charge()
        self.size = beam.get_size(mask)
        self.betagamma = beam.get_betagamma(mask)
        self.betalambda = beam.get_betalambda(mask)
        self.ref_phi = beam.get_ref_phi()
        self.ref_w = beam.get_ref_w()

    def get_ref_phi(self):
        """Return the phase of reference particle"""
        return self.ref_phi

    def get_ref_w(self):
        """Return the phase of reference particle"""
        return self.ref_w

    def get_current(self):
        """Return beam frequency in MHz"""
        return self.current

    def get_frequency(self):
        """Return beam frequency in MHz"""
        return self.frequency

    def get_mass(self):
        """Return mass of beam, mc^2, in MeV"""
        return self.mass

    def get_charge(self):
        """Return charge of beam in q/|e|"""
        return self.charge

    def get_betagamma(self):
        """Return betagamma of beam"""
        return self.betagamma

    def get_betalambda(self):
        """Return betalambda"""
        return self.betalambda

    def get_size(self):
        """Returns total number of beam macro particles"""
        return self.size

    def get_coor(self, var):
        if var in (_COORDINATES + _LOSSES):
            ndx = Distribution.coor_index[var]
            return self.coor[ndx]
        else:
            print "Error: Empty masked returned, variable", \
            str.upper(var), "not recognized"
            return np.array([])

    def get_loss_num(self):
        """Returns number of macro-particles lost transversely"""
        tloss = self.get_coor('losses')
        return len(tloss[tloss > 0])

    def get_avg(self, var):
        return np.mean(self.get_coor(var))

    def get_sig(self, var):
        return np.std(self.get_coor(var))

    def get_twiss(self, var):
        """Return Twiss parameters (a,b,unnormalized, rms e) for specified coords x, y or z"""
        lc_var = var.lower()
        if lc_var in _EMITTANCE:
            v = lc_var
            vp = v + 'p'
            if v == 'z':
                v = 'phi'
                vp = 'w'
            avgv2 = math.pow(self.get_sig(v), 2)
            avgvp2 = math.pow(self.get_sig(vp), 2)
            avgvvp = np.mean((self.get_coor(v) - self.get_avg(v)) *
                               (self.get_coor(vp) - self.get_avg(vp)))
            ermssq = avgv2 * avgvp2 - avgvvp * avgvvp
            if ermssq > 0:
                erms = math.sqrt(ermssq)
                alpha = - avgvvp / erms
                beta = avgv2 / erms
                return [alpha, beta, erms]            
            else:
                #print "Error: " + str.upper(var) + " emittance undefined"
                return 3*[float('NaN')]
        else:
            print "Error: Requested coordinate, " + str.upper(var) + \
                ", must be one of the following:", _EMITTANCE
            return 3*[float('NaN')]       

    def get_urms_emit(self, var):
        """Return unnormalized rms emittance along specified axis, x, y or z"""
        a, b, e_urms = self.get_twiss(var)
        return e_urms

    def get_nrms_emit(self, var):
        """Return normalized rms emittance along specified axis, x, y or z"""
        if var in ['x', 'y']:
            return self.get_urms_emit(var) * self.get_betagamma()
        else:
            return self.get_urms_emit(var)

    def print_results(self):
        """Prints avg, sigma, alpha, beta, Eurms, Enrms for all coord of distr"""
        print "Distribution Analysis Results (w/user units)"
        print 'Mass = {0:.4f}'.format(self.get_mass())
        print 'Charge/|e| = {0:.0f}'.format(self.get_charge())
        print 'Ib = {0:.2f} {1}'.format(self.get_current(), _USER_LABELS['i'])
        print 'Number of macroparticles = {0:.0f}'.format(self.get_size())
        print 'Frequency = {0:.3f} MHz'.format(self.get_frequency())
        print '*** Mask may have been applied to create Distribution object ***'
        print 'Number of macroparticles(in distrubution object) = {0:.0f}'.format(self.get_size())
        print 'Number of macroparticles lost (in distribution object) = {0:.0f}'.format(self.get_loss_num())
        print 'Ref part. \n phi = {0:10.4f} {1}\n   w = {2:10.4f} {3}'.\
            format(self.get_ref_phi(), _USER_LABELS['phi'],
                   self.get_ref_w(), _USER_LABELS['w'])
        if self.get_size() > 0:
            print '\nCentroids and RMS sizes'
            print '            Avg         Sigma'
            for item in _COORDINATES:
                print '{0:3}: {1:10.4f}    {2:10.4f} {3:3}'\
                    .format(item, self.get_avg(item),
                            self.get_sig(item), _USER_LABELS[item])
            print '\nTwiss parameters'
            print '         Alpha      Beta      Eurms      Enrms'
            for item in _EMITTANCE:
                a, b, eurms = self.get_twiss(item)
                print '{0:2}: {1:10.4f} {2:10.4f} {3:10.4f} {4:11.5f}'\
                    .format(item, a, b, eurms, self.get_nrms_emit(item))
            print '\n'
        else:
            print '\n*** No particles remaining ****'
            print '\n'
        return 
        
################################################################################

class DBConnection():
    """An hpsim class for creating the database connection.
    The user must provide the following arguments to constructor:

    databases: an ordered python list containing the individual database 
               filenames to be used in the simulations. The database must be 
               ordered to represent the linac from upstream to downstream
    """
    dbconnection = ""
# functions from original HPSim.so 

    def __init__(self, db_dir, databases, libsql_dir, libsql_file):
        """Init loads and attaches databases so those original functions are not
        separately available
        
        Arguments:
           db_dir (str): path of dir containing db files
           databases (list of str): ordered list of database filenames in correct sequence
           libsql_dir (str): path of directory that contains external sql lib
           libsql_file (str): name of libsqliteext.so file

        Returns:
           dbconnection object
        """
        self.dbconnection = ""
        db_num = 0
        db_name = "main"
        for db in databases:
            db_path = os.path.join(os.path.abspath(db_dir),db)
            if db is databases[0]:
                self.dbconnection = HPSim.DBConnection(db_path)
            else:
                db_num += 1
                db_name = 'db' + str(db_num)
                self.dbconnection.attach_db(db_path, db_name)
        # assign to class variable the HPSim DBConnection object for use elsewhere
        DBConnection.dbconnection = self.dbconnection
        libsql_path = os.path.join(os.path.abspath(libsql_dir), libsql_file)
        self.dbconnection.load_lib(libsql_path)
        self.clear_model_index()

    def print_dbs(self):
        """Prints names of datatbases"""
        self.dbconnection.print_dbs()

    def print_libs(self):
        """Prints the database library"""
        self.dbconnection.print_libs()

    def clear_model_index(self):
        """Clears model index. Must be called once db connection established"""
        self.dbconnection.clear_model_index()

    def get_epics_channels(self):
        """Returns a list of all the EPICS PV's in the db's connected thru 
        dbconnection"""
        return self.dbconnection.get_epics_channels()
            
################################################################################

class BeamLine():
    """An hpsim class for defining and accessing the beamline
    """
    beamline = ""
    def __init__(self):
        """
        Arguments: none
        Returns:
           beamline object
        """
        self.beamline = HPSim.BeamLine(DBConnection.dbconnection)
        # assign class variable with beamline object for use later
        BeamLine.beamline = self.beamline

# functions from original HPSim.so 
    def print_out(self):
        """Print complete beamline listing from pinned memory, element by element
        for benchmarking"""
        self.beamline.print_out()

    def print_range(self, start_element, end_element):
        """Print range of elements in pinned memory from start to end,
        for benchmarking

        Arguments:
           start_element (str): first element name in range to print
           last_element (str): last element name in range to print

        """
        self.beamline.print_range(start_element, end_element)

    def new_get_element_names(self, start_element=None, end_element=None, elem_type=None):
        """Get list of elements in beamline from start_element to end_element
        with option elem_type
        
        Arguments:
           start_element(str): first element to retrieve from beamline
           end_element(str): last element to retrieve from beamline
           elem_type(str, optional): type of element, e.g. 'WS' to retrieve

        Return:
           Python list of element names

        """
        BeamLine.beamline.get_element_names()#start_element, end_element, elem_type)

################################################################################

class SpaceCharge():
    """An hpsim class for defining and modifying the space charge used in 
    the simulation"""
    spacecharge = ""
# functions from original HPSim.so

    def __init__(self, nr, nz, interval, adj_bunch, type="'scheff'"):
        """Creates an instance of the space-charge class
        
        Arguments:
           nr (int): number of space-charge mesh slices in r direction
           nz (int): number of space-charge mesh slices in z direction
           interval (double): maximum spacing between space charge kicks
           adj_bunch (int): number of adj_bunch used in s.c. calc
           type (str, optional): "scheff" by default and is the only option 
                                  at the moment
        """
        request = "HPSim.SpaceCharge("
        request += "nr = " + str(nr)
        request += ", nz = " + str(nz)
        request += ", interval = " + str(interval) #in meters, not user units
        request += ", adj_bunch = " + str(adj_bunch)
#        request += ", type = " + type
        request += ")"
        #print request
        self.spacecharge = eval(request)
        SpaceCharge.spacecharge = self.spacecharge

    def get_interval(self):
        """Returns the interval, i.e. maximum drift distance (m) between 
        space-charge kicks"""
        return self.spacecharge.get_interval()
    
    def set_interval(self, interval):
        """Set maximum drift distance (m) between space-charge kick

        Argument:
           interval (double): maximum distance between space-charge kicks
        """
        self.spacecharge.set_interval(interval)

    def get_adj_bunch(self):
        """Return the number of adjacent bunches in space charge calculation"""
        return int(self.spacecharge.get_adj_bunch())

    def set_adj_bunch(self, adj_bunch):
        """Set the number of adjacent bunches used in space charge calculation
        Argument:
           adj_bunch (int): number of adjacent bunches to use in s.c. calc.
        """
        self.spacecharge.set_adj_bunch(int(adj_bunch))

    def get_adj_bunch_cutoff_w(self):
        """Return the cutoff energy (MeV) above which the adjacent bunches are
        no longer used in space charge calculation and s.c. mesh region based 
        upon beam size, i.e. 3*sigmas. This enables automatic transition to 
        faster s.c. calc once adjacent bunches need no longer be considered"""
        return self.spacecharge.get_adj_bunch_cutoff_w()

    def set_adj_bunch_cutoff_w(self, w_cutoff):
        """Set the cutoff energy (MeV) above which the adjacent bunchss are
        no longer used in space charge calculation and s.c. mesh region based 
        upon beam size, i.e. 3*sigmas. This enables automatic transition to 
        faster s.c. calc once adjacent bunches need no longer be considered

        Argument:
           w_cutoff (double): threshold energy above which adjacent bunches are no
                              longer used in s.c. calc
        """

        self.spacecharge.set_adj_bunch_cutoff_w(w_cutoff)

    def get_mesh_size(self):
        """Return a list of floats representing the r,z mesh size"""
        return self.spacecharge.get_mesh_size()

    def set_mesh_size(self, nr, nz):
        """Set the size of the mesh, i.e. nr, nz
        
        Arguments:
           nr (double): number of radial grid points
           nz (double): number of longitudinal grid points
        """
        self.spacecharge.set_mesh_size(nr, nz)

    def get_mesh_size_cutoff_w(self):
        """Return the cutoff energy for the beam at which the mesh size will
        decrease by nr/2 and nz/2 and interval increase by 4.This enables 
        automatic transition to faster s.c. calc."""
        return self.spacecharge.get_mesh_size_cutoff_w()

    def set_mesh_size_cutoff_w(self, w_cutoff):
        """Set the cutoff energy for decreasing the mesh by nr/2, nz/2 and 
        increasing interval by 4. This enables automatic transition to 
        faster s.c. calc.
        
        Arguments:
           w_cutoff (double): Threshold energy (MeV) where s.c. calc reduces 
                              nr by factor 2, nz by factor 2 and interval by factor 4.
        """
        self.spacecharge.set_mesh_size_cutoff_w(w_cutoff)

    def get_remesh_threshold(self):
        """Get the remeshing factor (default is 0.05) where
        0 => remesh before every space-charge kick
        >0 => adaptive algorithm determines how much beam shape can change 
        before mesh must be redone"""
        return self.spacecharge.get_remesh_threshold(rm_thres)

    def set_remesh_threshold(self, rm_thres):
        """Set the remeshing factor (default is 0.05) where
        0 => remesh at before every space-charge kick
        >0 => adaptive algorithm determines how much beam shape can change 
        before mesh must be redone
        
        Arguments:
           rm_thres (double): the factor that determines if the s.c. grid is remeshed
                              or not.
        """

        self.spacecharge.set_remesh_threshold(rm_thres)

################################################################################

class Simulator():
    """An hpsim class for defining the simulator"""

# functions from original HPSim.so 

    def __init__(self, beam):
        """Creates an instance of the simulator class

        Arguments:
           beam (object): beam class object

        Returns:
           Simulator class object
        """
        request = "HPSim.Simulator(beam.beam, BeamLine.beamline, "
        request += "SpaceCharge.spacecharge)"
        self.sim = eval(request)

    def simulate(self, start_elem_name, end_elem_name):
        """Simulate from 'start' element to 'end' element, inclusive"""
        self.sim.simulate(start_elem_name, end_elem_name)

    def set_space_charge(self, state='off'):
        """Turn space charge on or off
        state (str, optional): "on", "off"(default)
        """
        self.sim.set_space_charge(state)

################################################################################

class BeamPlot():
    """An hpsim class for creating beam plots"""

    def __init__(self, nrow=1, ncol=1, hsize=None, vsize=None):
        """Creates and instance of a matplotlib figure

        Arguments:
           nrow (int): number of rows in figure plotting grid
           ncol (int): number of columns in figure plotting grid
           hsize (double): horizontal size (inches) of figure
           vsize (double): vertical size (inches) of figure
        """
        if hsize == None or vsize == None:
            self.fig = plt.figure(facecolor='white')
        else:
            self.fig = plt.figure(figsize = (hsize, vsize), facecolor='white')

        self.nrow = nrow
        self.ncol = ncol
        print '{0:2} x {1:2} BeamPlot object created'.format(self.nrow, self.ncol)

    def title(self, title):
        """Place title string in window bar
        Arguments:
           title (str): figure title
        """
        self.fig.canvas.set_window_title(title)

    def clear(self):
        """Clear plot figure"""
        self.fig.clf()

    def hist1d(self, u_vals, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create 1d histogram of arbitrary vals in numpy array

        Arguments:
           u_vals (Numpy vector):values to plot
           nplt (int):which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (optional, [list of doubles]) [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False(default)-> linear plot

        """
        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
        _plot_hist1d(plt, u_vals, nplt, nbins, xlabel, limits, norm, ylog)
        return

    def hist1d_coor(self, coor, beam, mask, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create a histogram style profile of beam coordinate

        Arguments:
           coor (str): coordinate to plot, either 'x', 'xp', 'y', 'yp', 'phi', 'w' or 'losses'
           beam (beam object): beam object containing coordinates to plot
           mask (numpy vector): mask for filter beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False-> linear plot
        """

        if coor in _COORDINATES:
            u_index = _COORDINATES.index(coor)
            u_label = _COORDINATES[u_index]
            u_coor = beam.get_coor(u_label, mask)
            label = u_label + ' ' + _USER_LABELS[u_label]

        elif coor in _LOSSES:
            u_coor = 1.0*beam.get_coor('losses', mask)
            label = 'losses along beamline'

        if xlabel is not None:
            label = xlabel

        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
        _plot_hist1d(plt, u_coor, nplt, nbins, label, limits, norm, ylog)

        return

    def profile(self, coor, beam, mask, nplt, marker='g-', nbins=50, limits=None, ylog=False):
        """Create a profile of beam coordinate 

        Arguments:
           coor (str): coordinate to plot, either 'x', 'xp', 'y', 'yp', 'phi', 'w' or 'losses'
           beam (beam object): beam object containing coordinates to plot
           mask (numpy vector): mask for filter beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           marker (str, optional): matplotlib color and marker, e.g. 'r.'
           nbins (int, optional): number of bins to plot
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
           ylog (logical, optional): True-> semilog plot, False-> linear plot
        """ 
        if coor in _COORDINATES:
            u_index = _COORDINATES.index(coor)
            u_label = _COORDINATES[u_index]
            u_coor = beam.get_coor(u_label, mask)
            label = u_label + ' ' + _USER_LABELS[u_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
            _plot_profile(plt, u_coor, marker, nbins, label, limits, ylog)

        return

    def phase_space(self, coor, beam, mask, nplt, marker='b,', limits=None):
        """Create beam phase space dot plot as nth subplot to figure

        Arguments:
           coor (str): text string either 'xxp', 'yyp' or 'phiw'
           beam (object): object containing beam to be plotted
           mask (Numpy array): mask for filter beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                        1 is upper left, nrow*ncol is lower right
           marker (str, optional): matplotlib color and marker, e.g. 'r.'
           limits (list of doubles, optional): plot limits [[xmin, xmax], [ymin, ymax]]
        """
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index]
            v_label = _COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)

            labels=['','']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
            _plot_phase_space(plt, u_coor, v_coor, marker, labels, limits)
        return
        
    def iso_phase_space(self, coor, beam, mask, nplt, nbins=[50,50]):
        """Create an isometric phase-space plot.

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           beam (beam object): beam object containing coordinates to plot
           mask (Numpy vector): mask for filtering beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins ([int, int], optional): number of x and y bins, respectively
        """
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index] 
            v_label = _COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)

            labels=['','']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt, projection='3d')
            _plot_iso_phase_space(plt, u_coor, v_coor, labels, nbins)

        return

    def surf_phase_space(self, coor, beam, mask, nplt, nbins=100, limits=None):
        """Create a surface phase-space plot

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           beam (beam object): beam object containing coordinates to plot
           mask (Numpy vector): mask for filtering beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """
        #nbins = 50
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index]
            v_label = _COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)

            labels=['','']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            ax = self.fig.add_subplot(self.nrow, self.ncol, nplt, projection='3d')

            _plot_surf_phase_space(ax, u_coor, v_coor, labels, nbins, limits)
        return

    def hist2d(self, u_vals, v_vals, nplt, labels=None, nbins=100, limits=None):
        """Create an 2d histogram of user given u & v values

        Arguments:
           u_vals (Numpy vector):values to plot u-axis
           v_vals (Numpy vector):values to plot v-axis
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           labels ([str, str]): u- and v-axes lables
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """

        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)          
        _plot_hist2d(plt, u_vals, v_vals, labels=labels, nbins=nbins, limits=limits)

        return

    def hist2d_phase_space(self, coor, beam, mask, nplt, nbins=100, limits=None):
        """Create an 2d histogram phase-space plot

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           beam (beam object): beam object containing coordinates to plot
           mask (Numpy vector, int): mask for filtering beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index]
            v_label = _COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)
        
            labels = ['', '']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)          
            _plot_hist2d(plt, u_coor, v_coor, labels=labels, nbins=nbins, limits=limits)

        return

    def draw(self):
        """Draw figure. Used in interactive mode"""
        plt.draw()
        return

    def show(self):
        """Show the plots. Used in non-interactive mode"""
        plt.tight_layout()
        plt.show()
        return
################################################################################

class DistPlot():
    """An hpsim class for creating plots of beam distribution objects"""

    def __init__(self, nrow=1, ncol=1, hsize=None, vsize=None):
        """Creates and instance of a matplotlib figure"""
        if hsize == None or vsize == None:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure(figsize = (hsize, vsize))

        self.nrow = nrow
        self.ncol = ncol
        print '{0:2} x {1:2} DistPlot object created'.format(self.nrow, self.ncol)

    def title(self, title):
        """Place title string in window bar"""
        self.fig.canvas.set_window_title(title)

    def clear(self):
        """Clear plot figure"""
        self.fig.clf()

    def hist1d(self, u_vals, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create 1d histogram of arbitrary vals in numpy array

        Arguments:
           u_vals (Numpy vector):values to plot
           nplt (int):which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (optional, [list of doubles]) [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False(default)-> linear plot

        """
        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
        _plot_hist1d(plt, u_vals, nplt, nbins, xlabel, limits, norm, ylog)
        return

    def hist1d_coor(self, coor, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create a histogram style profile of beam coordinate

        Arguments:
           coor (str): coordinate to plot, either 'x', 'xp', 'y', 'yp', 'phi', 'w' or 'losses'
           dist is beam-distribution object
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False-> linear plot
        """

        if coor in _COORDINATES:
            u_index = _COORDINATES.index(coor)
            u_label = _COORDINATES[u_index]
            u_coor = dist.get_coor(u_label, mask)
            label = u_label + ' ' + _USER_LABELS[u_label]

        elif coor in _LOSSES:
            u_coor = 1.0 * dist.get_coor('losses', mask)
            label = 'losses along beamline'

        if xlabel is not None:
            label = xlabel

        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
        _plot_hist1d(plt, u_coor, nplt, nbins, label, limits, norm, ylog)

        return

    def profile(self, coor, dist, nplt, marker='g-', nbins=50, limits=None, ylog=False):
        """Add profile of beam coordinate 
        Arguments:
           coor (str): coordinate to plot, either 'x', 'xp', 'y', 'yp', 'phi', 'w' or 'losses'
           dist is beam-distribution object
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           marker (str, optional): matplotlib color and marker, e.g. 'r.'
           nbins (int, optional): number of bins to plot
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
           ylog (logical, optional): True-> semilog plot, False-> linear plot
        """ 
        if coor in _COORDINATES:
            u_index = _COORDINATES.index(coor)
            u_label = _COORDINATES[u_index]
            u_coor = dist.get_coor(u_label)
            label = u_label + ' ' + _USER_LABELS[u_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
            _plot_profile(plt, u_coor, marker, nbins, label, limits, ylog)

        return

    def phase_space(self, coor, dist, nplt, marker='b,', limits=None):
        """Add beam phase space as nth subplot to figure

        Arguments:
           coor (str): text string either 'xxp', 'yyp' or 'phiw'
           dist is beam-distribution object
           nplt (int): which plot in figure grid, by row, 
                        1 is upper left, nrow*ncol is lower right
           marker (str, optional): matplotlib color and marker, e.g. 'r.'
           limits (list of doubles, optional): plot limits [[xmin, xmax], [ymin, ymax]]
        """
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index]
            v_label = _COORDINATES[v_index]
            u_coor = dist.get_coor(u_label)
            v_coor = dist.get_coor(v_label)

            labels=['','']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
            _plot_phase_space(plt, u_coor, v_coor, marker, labels, limits)
        return
        
    def iso_phase_space(self, coor, dist, nplt, nbins=[50,50]):
        """Create an isometric phase-space plot.

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           dist is beam-distribution object
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins ([int, int], optional): number of x and y bins, respectively
        """
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index] 
            v_label = _COORDINATES[v_index]
            u_coor = dist.get_coor(u_label)
            v_coor = dist.get_coor(v_label)

            labels=['','']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt, projection='3d')
            _plot_iso_phase_space(plt, u_coor, v_coor, labels, nbins)

        return

    def surf_phase_space(self, coor, dist, nplt, nbins=100, limits=None):
        """Create a surface phase-space plot

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           dist is beam-distribution object
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """
        #nbins = 50
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index]
            v_label = _COORDINATES[v_index]
            u_coor = dist.get_coor(u_label, mask)
            v_coor = dist.get_coor(v_label, mask)

            labels=['','']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            ax = self.fig.add_subplot(self.nrow, self.ncol, nplt, projection='3d')

            _plot_surf_phase_space(ax, u_coor, v_coor, labels, nbins, limits)
        return

    def hist2d(self, u_vals, v_vals, nplt, labels=None, nbins=100, limits=None):
        """Create an 2d histogram of user given u & v values

        Arguments:
           u_vals (Numpy vector):values to plot u-axis
           v_vals (Numpy vector):values to plot v-axis
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           labels ([str, str]): u- and v-axes lables
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """

        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)          
        _plot_hist2d(plt, u_vals, v_vals, labels=labels, nbins=nbins, limits=limits)

        return

    def hist2d_phase_space(self, coor, dist, nplt, nbins=100, limits=None):
        """Create an 2d histogram phase-space plot

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           dist is beam-distribution object
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """
        if coor in _PHASESPACE:
            u_index = _PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = _COORDINATES[u_index]
            v_label = _COORDINATES[v_index]
            u_coor = dist.get_coor(u_label)
            v_coor = dist.get_coor(v_label)
        
            labels = ['', '']
            labels[0] = u_label + ' ' + _USER_LABELS[u_label]
            labels[1] = v_label + ' ' + _USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)          
            _plot_hist2d(plt, u_coor, v_coor, labels=labels, nbins=nbins, limits=limits)

        return

    def draw(self):
        """Draw figure. Used in interactive mode"""
        plt.draw()
        return

    def show(self):
        """Show the plots"""
        plt.tight_layout()
        plt.show()
        return
################################################################################

class DBState():
    """An hpsim class capturing the state of all EPICS PV's in the connected dB's"""

    def __init__(self):
        """Create dB state object"""
        self.db_array=[]

    def get_db_pvs(self, file = None):
        """Record all Epics PV names and values from db 
        If filename present then also write to that file

        Arguments:
           file (str, optional): filename to write output to

        """
        self.epics_pvs = DBConnection.dbconnection.get_epics_channels()
        db_array = []
        for pv in self.epics_pvs:
            val = get_db_epics(pv)
            db_array.append([pv, val])
        self.db_array = db_array
        print "*** dB PV's stored in state object ***"
        if file is not None:
            # store db_pvs in file
            fid = open(file,'w')
            fid.write(self.db_array)
            fid.close()
            print "*** dB PV's written to file: ", file, " ***"
            
    def restore_db_pvs(self, file = None):
        """Restore EPICS PV in file or DBState object back into dB
        If file present use file else use DBState object

        Arguments:
           file (str, optional): filename from which to extract dB Epics PV values

        """
        if file is not None:
            # restore from file
            fid = open(file,'r')
            loc_db_array = fid.read()
            fid.close()
            for item in self.db_array:
                pv, val = item
                set_db_epics(pv, val)
                print "*** dB PV's restored from file: ", file, " ***"
        else:
            # restore from db_array object
            for item in self.db_array:
                pv, val = item
                set_db_epics(pv, val)
                print "*** dB PV's restored from db_array object ***"

    def print_pvs(self, pvname = None):
        """Print vals of EPICS PVs in DBState object that correspond to pvname
        Print all PVs vals in state object if pvname is not supplied 
        
        Arguments:
           pvname (str, optional): print value of named Epics PV

        """
        print '*** PV vals in state object ***'
        if pvname is not None:
            loc_pv = lcs.expand_pv(pvname)
        for item in self.db_array:
            pv, val = item
            if pvname is None:
                print '{0} = {1}'.format(pv, val)
            elif pv[0:len(loc_pv)] == loc_pv:
                print '{0} = {1}'.format(pv, val)

    def turn_off(self, pv_name):
        """Set all PV's with name pv_name to val of zero
        
        Arguments:
           pv_name(str): name of Epics PV

        """
        loc_pv = lcs.expand_pv(pv_name)
        for item in self.db_array:
            pv, val = item
            if pv[0:len(loc_pv)] == loc_pv:
                set_db_epics(pv, 0.0)
        
    def turn_on(self, pv_name):
        """Restore all PV's with name to associated vals from DBState
        
        Arguments:
           pv_name(str): name of Epics PV
        """
        loc_pv = lcs.expand_pv(pv_name)
        for item in self.db_array:
            pv, val = item
            if pv[0:len(loc_pv)] == loc_pv:
                set_db_epics(pv, val)
        
################################################################################
#
# miscellaneous functions
#

def set_gpu(n):
    """Set which GPU to use
    Arguments:
       n(int): number of GPU to use
       
    """
    HPSim.set_gpu(n)

def most_freq_value(vals):
    """ Returns an estimate of the most frequently occuring value
    by first histograming the the Numpy array npvals in a histogram
    with unit bins, then finding the peak, then averaging that along 
    with the adjacent bins to get an estimate of the value that 
    represents the most frequency value.

    Arguments:
       vals(Numpy array): input 1D array

    Returns:
       estimate of the value that occurs most frequently

    """
    minval = min(vals)
    maxval = max(vals)
    bins = max(int(maxval-minval) + 1, 3)
    hist, bin_edge = np.histogram(vals, bins=bins, density=False)
    nbins = len(hist)
    bin_width = bin_edge[1] - bin_edge[0]
    hist_max = np.amax(hist)
    bin_max_indx = np.argmax(hist)
    if bin_max_indx == 0:
        pk_bin_avg = bin_edge[bin_max_indx]
    elif bin_max_indx > 0 and bin_max_indx < nbins:
        ll = bin_max_indx - 1
        ul = bin_max_indx + 2
        pk_bin_avg = np.average(bin_edge[ll:ul], weights=hist[ll:ul])
    else:
        pk_bin_avg = bin_edge[bin_max_indx]

    return pk_bin_avg + 0.5 * bin_width

def modulo_phase(phase_dist, ref_phs):
    """Return the phase coordinates of beam after modulo 360 deg wrt ref_phs 
       has been applied

    Arguments:

       phase_dist (Numpy vector, doubles): phase coordinates (deg)
       ref_phs (double): reference phase for modulo calc

    """

    return ((phase_dist - ref_phs + math.pi * _DEG) % (2*math.pi * _DEG)) + ref_phs - (math.pi * _DEG)
    
def set_db_epics(pv_name, value):
    """Change EPICS PV value in database

    Arguments:

       pv_name(str): EPICS pv name string
       value(str or double): value to set Epics PV to 
       Note: DBConnection and BeamLine must be already be established
    """
    HPSim.set_db_epics(pv_name, str(value), DBConnection.dbconnection,
                     BeamLine.beamline)

def get_db_epics(pv_name):
    """Retrieve EPICS PV value in database

    Arguments:
       pv_name (str): EPICS pv name string
       Note: DBConnection must be already be established
    """
    value = HPSim.get_db_epics(pv_name, DBConnection.dbconnection)
    if lcs.get_pv_type(pv_name) is not 'L':
        value = float(value)
    return value
    
def set_db_model(table_name, field_name, value):
    """Change model database parameter value given by table_name and field_name
    
    Arguments:
       table_name (str): name of element in db table
       field_name (str): name of field to change of element in table
       value (str or double): value to set db field to 
       Note: DBConnection and BeamLine must be already be established
    """
    HPSim.set_db_model(table_name, field_name, str(value), DBConnection.dbconnection,
                     BeamLine.beamline)

def get_db_model(elem_name, field_name):
    """Retrieve model database parameter value given by table_name and field_name

    Arguments:
       table_name (str): name of element in db table
       field_name (str): name of field to change of element in table
       Note: DBConnection and BeamLine must be already be established
    """
    text_fields = ['id', 'name', 'model_type']
    value = HPSim.get_db_model(elem_name, field_name, DBConnection.dbconnection)
    if field_name not in text_fields:
        #convert to float
        value = float(value)
    return value
        
def get_beamline_direction(start_elem, stop_elem):
    """Returns +1 for stop_elem beyond start_elem or -1 if stop_elem behind start_elem
    
    Arguments:
       start_elem(str): beginning element name
       stop_elem(str): final element name
    """
    direction = 1
    beamline = get_element_list()
    istart = beamline.index(start_elem)
    istop = beamline.index(stop_elem)
    if istart > istop:
        direction = -1
    return direction    

def get_element_list(start_elem_name = None, end_elem_name = None, elem_type = None):
    """Retrieve a list containing the names of beamline elements from 
    'start_elem_name' to 'end_elem_name'

    Arguments:
       start_elem_name(str): first element in list
       end_elem_name(str): last element in list
       elem_type(str): type of element (db type or C++ type) to retrieve

    """
    elem_type_dict = {'caperture':'ApertureC', 'raperture':'ApertureR',\
                        'buncher':'Buncher', 'diagnostics':'Diagnostics',\
                        'dipole':'Dipole', 'drift':'Drift', 'quad':'Quad',\
                        'dtl-gap':'RFGap-DTL', 'ccl-gap':'RFGap-CCL',\
                        'rotation':'Rotation', 'spch_comp':'SpchComp'}

    if elem_type in elem_type_dict.keys():
        elem_type_resolved = elem_type_dict[elem_type]
    else:
        elem_type_resolved = elem_type

    request = "BeamLine.beamline.get_element_names("
    if start_elem_name:
        request += '"' + start_elem_name + '"'
    else:
        request += '""'
    if end_elem_name:
        request += ', ' + '"' + end_elem_name + '"'
    else:
        request += ', ' + '""'
    if elem_type:
        request += ', '
        request += '"' + elem_type_resolved + '"' 
    request += ')'
#    print request
    return eval(request)

def get_element_length(elem_name):
    """Return length of beamline element in hpsim base units(m).
    
    Arguments:
       elem_name(str): name of element

    """
    elem_type = get_db_model(elem_name, 'model_type')
    eff_len = 0.0
    if elem_type in ['drift', 'quad', 'dtl_gap', 'ccl_gap']:
        # elements with a defined length
        eff_len = get_db_model(elem_name, 'length_model')
    elif elem_type in ['dipole']:
        # effective path length must be calculated
        eff_len = get_db_model(elem_name, 'rho_model') \
            * get_db_model(elem_name, 'angle_model')
    return eff_len

def get_beamline_length(start, end):
    """Returns length of beamline from element 'start' to element 'end'
    in hpsim base units (m). If start is after stop, then the length is < 0

    Arguments:
       start(str): first element in list
       end(str): last element in list

    """
    l = 0.0
    direction = get_beamline_direction(start, end)
    if direction < 0:
        loc_start, loc_end = end, start
    else:
        loc_start, loc_end = start, end
    bl = get_element_list(start_elem_name = loc_start, end_elem_name = loc_end)
    for elem in bl:
        #print elem, get_element_length(elem)
        l += get_element_length(elem)
    return l * direction

def get_beamline_midpoints():
    """Returns a list of the distance to the midpoint of each element in 
    the complete beamline, units (m). 

    Arguments: None
    """
    bl_length = 0.0
    bl = get_element_list()
    midpoints = []
    for i,elem in enumerate(bl):
        elem_length = get_element_length(elem)
        midpoints.append(bl_length + 0.5 * elem_length)
        bl_length += elem_length
        print i,elem, elem_length, midpoints[-1], bl_length
    return

def get_first_element():
    """Returns name of first element in connected database

    Arguments: None
    """
    return get_element_list()[0]

def get_last_element():
    """Returns name of first element in connected database

    Arguments: None
    """
    return get_element_list()[-1]

def get_next_element(elem_name):
    """Returns the name of the next element in the connected databases
    Arguments:
       elem_name(str): name of element
    """
    beamline = get_element_list()
    if elem_name in beamline:
        next = beamline.index(elem_name) + 1
        if next < len(beamline):
            return beamline[next]
        else:
            return None
    else:
        print elem_name, "not found in beamline list"

def get_mmf(twiss1, twiss2):
    """Returns the MisMatch Factor between to sets of Twiss parameters
    where Twiss is (alpha, beta, eps)

    Arguments:
       twiss1(list of doubles): [alpha, beta, emittance]
       twiss2(list of doubles): [alpha, beta, emittance]
    """
    a1, b1, e1 = twiss1
    g1 = (1.0 + a1 * a1) / b1
    a2, b2, e2 = twiss2
    g2 = (1.0 + a2 * a2) / b2
    r = b1 * g2 + g1 * b2 - 2.0 * a1 * a2
    mmf = math.sqrt(0.5 * (r + math.sqrt(r * r - 4))) - 1.0
    return mmf

def betalambda(mass, freq, w):
    """Return value of beta*lambda of beam.

    Arguments:
       mass(double): mc^2 of beam particle in MeV
       freq(double): frequency in MHz
       w(double): Kinetic energy in MeV
    """
    c = 2.99792458e8 # m/s
    wavelength = c / (freq * 1.0e6)
    gamma = 1.0 + w / mass
    beta = math.sqrt(1.0 - 1/(gamma * gamma))
    return beta * wavelength
    
################################################################################
#
# private  functions
#

def _get_labels(labels):
    """Get axis labels for plotting
    Arguments:
       labels([str, str]): list of x, y-axis labels
    """
    if labels == None:
        u_label = 'x-axis'
        v_label = 'y-axis'
    elif isinstance(labels, list):
        u_label = labels[0]
        v_label = labels[1]
    else:
        u_label = ''
        v_label = ''
    return (u_label, v_label)

def _get_plimits(limits, u_coor, v_coor):
    """Returns the xmin, xmax, ymin, ymax for plot range.

    Private method

    Arguments:
       plim (list of doubles): [[xmin, xmax],[ymin, ymax]] or 
                               [[xmin, xmax],[]] or [[], [ymin, ymax]] or 
                               [ymin, ymax] or None
       u_coor (Numpy vector doubles): x_coordinates
       v_coor (Numpy vector doubles): y_coordinates

    Returns:
       list (double): [[xlo, xup],[ylo, yup]] containing the x and y limits
                      for the plotting range
    """

    if limits not in [None, []]:
        if isinstance(limits[0], list):
            #list of lists [[xmin, xmax],[ymin, ymax]]
            if limits[0] <> []:
                min_x = limits[0][0]
                max_x = limits[0][1]
            else:
                min_x = min(u_coor)
                max_x = max(u_coor)
                if min_x == max_x:
                    min_x = u_coor *0.9
                    max_x = u_coor *1.1

            if limits[1] <> []:
                min_y = limits[1][0]
                max_y = limits[1][1]
            else:
                min_y = min(v_coor)
                max_y = max(v_coor)
                if min_y == max_y:
                    min_y = v_coor *0.9
                    max_y = v_coor *1.1

        else:
            # list of y-values only, [ymin, ymax]
            min_x = min(u_coor)
            max_x = max(u_coor)
            min_y = limits[0]
            max_y = limits[1]
    else:
        min_x = min(u_coor)
        max_x = max(u_coor)
        min_y = min(v_coor)
        max_y = max(v_coor)
        if min_x == max_x:
            min_x = u_coor *0.9
            max_x = u_coor *1.1
        if min_y == max_y:
            min_y = v_coor *0.9
            max_y = v_coor *1.1

    return (min_x, max_x), (min_y, max_y)


def _plot_hist1d(plt, u_vals, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create 1d histogram of arbitrary vals in numpy array

        Private method

        Arguments:
           plt (Pyplot figure subplot object): figure to place subplot in
           u_vals (Numpy vector):values to plot
           nplt (int):which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (list of doubles, optional): [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False(default)-> linear plot

        """
        wghts = (1.0 / norm) * np.ones(len(u_vals))
        hist, bins, patches = plt.hist(u_vals, bins=nbins, weights=wghts, log=ylog)
        xrng, yrng = _get_plimits(limits, bins, hist)
        if xlabel is None:
            xlabel = 'variable'
        plt.set_xlabel(xlabel)
        plt.set_ylabel('counts/bin')
        if limits is not None:
            plt.set_xlim(xrng)
            if ylog is False:
                plt.set_ylim(yrng)
        return

def _plot_profile(plt, u_vals, marker='g-', nbins=50, label=None, limits=None, ylog=False):
    """Create a profile of beam coordinate 

    Arguments:
       plt (Pyplot plot object): plot figure object
       u_vals (Numpy vector, double): x-coordinates of data to be plotted
       marker (str, optional): matplotlib color and marker, e.g. 'r.'
       nbins (int, optional): number of bins to plot
       label (str): u-axis label
       limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
       ylog (logical, optional): True-> semilog plot, False-> linear plot
    """ 
    hist, bins = np.histogram(u_vals, bins = nbins)
    xrng, yrng = _get_plimits(limits, bins, hist)

    plt.set_xlabel(label)
    plt.set_ylabel('counts/bin')
    if ylog:
        plt.semilogy(bins[:-1], hist, marker)
        plt.set_xlim(xrng)
        if not (limits == None or limits[1] == []):
            plt.set_ylim(yrng)
    else:
        plt.plot(bins[:-1], hist, marker)
        plt.set_xlim(xrng)
        plt.set_ylim(yrng)

    return

def _plot_phase_space(plt, u_coor, v_coor, marker='b,', labels=None, limits=None):
    """Create beam phase space dot plot as nth subplot to figure

    Arguments:
       plt (Pyplot plot object): plot figure object
       u_coor (Numpy vector, double): x-coordinates of data to be plotted
       v_coor (Numpy vector, double): y-coordinates of data to be plotted
       marker (str, optional): matplotlib color and marker, e.g. 'r.'
       labels ([str, str]): u- and v-axes lables 
       limits (list of doubles, optional): plot limits [[xmin, xmax], [ymin, ymax]]
    """
    u_label, v_label = _get_labels(labels)
    xrng, yrng = _get_plimits(limits, u_coor, v_coor)
    plt.set_xlabel(u_label)
    plt.set_ylabel(v_label)
    plt.plot(u_coor, v_coor, marker)
    plt.set_xlim(xrng)
    plt.set_ylim(yrng)
    return

def _plot_iso_phase_space(ax, u_coor, v_coor, labels=None, nbins=[50,50]):
    """Create an isometric phase-space plot.

    Arguments:
       ax (Pyplot plot object): plot figure object
       u_coor (Numpy vector, double): x-coordinates of data to be plotted
       v_coor (Numpy vector, double): y-coordinates of data to be plotted
       labels ([str, str]): u- and v-axes lables 
       nbins ([int, int], optional): number of x and y bins, respectively
    """

    u_label, v_label = _get_labels(labels)
    ps_histo, u_bins, v_bins = np.histogram2d(y=v_coor, x=u_coor, bins=nbins) #\
#                                           range = [[min(u_coor), max(u_coor)],\
#                                                    [min(v_coor), max(v_coor)]])
    ps_histo = ps_histo.T

    verts = []
    for v_slice in ps_histo:
        v_slice[0] = 0.
        v_slice[-1] = 0.
        verts.append(list(zip(u_bins, v_slice)))

    poly = PolyCollection(verts, closed=False)

    from matplotlib import cm
#            m = cm.ScalarMappable(cmap=cm.jet)
#            m.set_clim(vmin=0, vmax=100)
    poly.set_cmap(cm.jet)
    poly.set_clim(vmin=0, vmax=100)
    poly.set_color('lightgreen')
#            poly.set_color('blue')
    poly.set_edgecolor('gray')
#            poly.set_edgecolor('white')
    poly.set_alpha(0.75)

    ax.add_collection3d(poly, zs=v_bins, zdir='y')

    ax.set_xlabel(u_label)
    ax.set_xlim3d(u_bins[0],u_bins[-1])
    ax.set_ylabel(v_label)
    ax.set_ylim3d(v_bins[0],v_bins[-1])
    ax.set_zlabel('Amplitude [counts]')
    ax.set_zlim3d(0, np.max(ps_histo))

    return

def _plot_surf_phase_space(ax, u_coor, v_coor, labels=None, nbins=100, limits=None):
    """Create a surface phase-space plot

    Arguments:
       ax (Pyplot plot object): plot figure object
       u_coor (Numpy vector, double): x-coordinates of data to be plotted
       v_coor (Numpy vector, double): y-coordinates of data to be plotted
       labels ([str, str]): u- and v-axes lables
       nbins (int, optional): number of x and y bins
       limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
    """

    u_label, v_label = _get_labels(labels)
    [min_x, max_x], [min_y, max_y] = _get_plimits(limits, u_coor, v_coor) 

    dx = (max_x - min_x)/float(nbins)
    dy = (max_y - min_y)/float(nbins)

    X = np.arange(min_x, max_x, dx)
    Y = np.arange(min_y, max_y, dy)
    X, Y = np.meshgrid(X, Y)

    ps_histo, u_bins, v_bins = np.histogram2d(y=v_coor, x=u_coor, bins=nbins, \
                                              range = [[min_x, max_x], [min_y, max_y]],\
                                              normed=False) #True)
    ps_histo = ps_histo.T

    ZMAX = float(np.max(ps_histo))
    colors = [[(0,0,0,0) for _ in range(nbins)] for _ in range(nbins)]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colormap = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
    for y in range(nbins):
        for x in range(nbins):
            if ps_histo[x, y] == 0:
                colors[x][y] = (1,1,1,1)
            else:
                colors[x][y] = colormap.to_rgba(ps_histo[x,y]/ZMAX)

    ps_surf = ax.plot_surface(X, Y, ps_histo, rstride=1, cstride=1, linewidth=0, facecolors=colors)

    ax.set_xlabel(u_label)
    ax.set_xlim3d(u_bins[0], u_bins[-1])
    ax.set_ylabel(v_label)
    ax.set_ylim3d(v_bins[0], v_bins[-1])
    ax.set_zlabel('Amplitude [counts]')
    ax.set_zlim3d(0, np.max(ps_histo))
    return

def _plot_hist2d(axPS, u_vals, v_vals, labels=None, nbins=100, limits=None):
    """Create an 2d histogram of user given u & v values

    Arguments:
       axPS (Pyplot figure object): plot object in figure
       u_vals (Numpy vector):values to plot u-axis
       v_vals (Numpy vector):values to plot v-axis
       labels ([str, str]): u- and v-axes lables
       nbins (int, optional): number of x and y bins
       limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
    """

    u_label, v_label = _get_labels(labels)

    # the scatter plot:

    [min_x, max_x], [min_y, max_y] = _get_plimits(limits, u_vals, v_vals) 

    ps_histo, u_bins, v_bins = np.histogram2d(y=v_vals, x=u_vals, bins=nbins, \
                                              range = [[min_x, max_x], [min_y, max_y]],\
                                              normed=True)
    ps_histo = ps_histo.T

    # mask zeros so they are not plotted
    ps_histo_masked = np.ma.masked_where(ps_histo == 0, ps_histo)

    extent = [min_x, max_x, min_y, max_y]
    figasp = float(max_x - min_x)/float(max_y - min_y)

    axPS.set_xlabel(u_label)
    axPS.set_ylabel(v_label)
    axPS.imshow(ps_histo_masked, extent=extent,interpolation='none', origin='lower', \
                    aspect=figasp)
    return
