import os
import sys

par_dir = os.path.abspath(os.path.pardir)
lib_dir = os.path.join(par_dir, "bin")
print lib_dir
sys.path.append(lib_dir)

import HPSim as hp

beam = hp.Beam(file="Hm_clz_64k.txt")
