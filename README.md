# HPSim

HPSim is a GPU-accelerated online multi-particle beam dynamics simulation tool 
for ion linacs. It was originally developed for use on the Los Alamos 800-MeV 
proton linac with the goal of providing a fast and accurate tool for realistic 
beam simulations of the actual operating linac. It runs on a GPU equipped 
workstation. The simulation parameters mirror those of the actual accelerator 
and can be manipulated through Python scripts in an offline mode or track them 
when connected via EPICS in an online mode.

It is a “z-code” that contains typical linac beam transport elements. The linac 
RF-gap transformation utilizes transit-time-factors to calculate the beam 
acceleration therein. The space-charge effects are computed using the 2D SCHEFF 
(Space CHarge EFFect) algorithm, which calculates the radial and longitudinal 
space charge forces for cylindrically symmetric beam distributions. Other 
space-charge routines to be incorporated include the 3D PICNIC and a 3D Poisson 
solver. HPSim can simulate beam dynamics in drift tube linacs (DTLs) and 
coupled cavity linacs (CCLs). Elliptical superconducting cavity (SC) structures 
will also be incorporated into the code.
 
The computational core of the code is written in C++ and accelerated using the 
NVIDIA CUDA technology. Users access the core code, which is wrapped in 
Python/C APIs, via Pythons scripts that enable ease-of-use and automation of 
the simulations. The overall linac description including the EPICS Process 
Variable machine control parameters is kept in an SQLite database that also 
contains calibration and conversion factors required to transform the machine 
set points into model values used in the simulation.

# Simulated H- beam in LANSCE LEBT + DTL (2D)
<a href="https://github.com/apphys/hpsim/blob/master/hpsim-2d.gif"><img src="https://github.com/apphys/hpsim/blob/master/hpsim-2d.gif" title="HPSim 2D"/></a>
# Simulated H- beam in LANSCE LEBT + DTL + CCL module 5 (3D)
<a href="https://github.com/apphys/hpsim/blob/master/hpsim-3d.gif"><img src="https://github.com/apphys/hpsim/blob/master/hpsim-3d.gif" title="HPSim 3D"/></a>

# Build & Run on Darwin (Power 8 + Tesla P100)
% module load cuda
% git checkout noepics
% cd src
% make

Now../bin/HPSim.so should be generated and we can run a test 

% cd ../pytest
% python sim-lbeg.py

# References
1. X. Pang, L. Rybarcyk, "GPU Accelerated Online Beam Dynamics Simulator for 
Linear Particle Accelerators," Computer Physics Communications, 185, pp. 
744-753, 2014.

2. X. Pang, "Advances in Proton Linac Online Modeling," Proceedings of the 6th 
International Particle Accelerator Conference, Richmond, VA, 2015.

# Copyright
Copyright (c) 2016, Los Alamos National Security, LLC
All rights reserved.
Copyright 2016. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
 
Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
3. Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

