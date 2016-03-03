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

conn = hp.DBConnection("../db/offline-dtl.db")
conn.load_lib("../db/lib/libsqliteext.so")
bl = hp.BeamLine(conn)
#bl.print_range('01RG01', '01RG02')
#hp.set_db_epics('01TM001L01', '0', conn, bl)
#bl.print_range('01RG01', '01RG02')

#print hp.get_element_list(bl, '01RG01', '', 'Drift')
print bl.get_element_names('01RG01', '', type='Quad')
