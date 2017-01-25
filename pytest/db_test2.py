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

#conn = hp.DBConnection("../db/offline-dtl.db")
#conn = hp.DBConnection("../db/offline-trst.db")
conn = hp.DBConnection("../db/offline-ccl-new.db")
conn.load_lib("../db/lib/libsqliteext.so")
bl = hp.BeamLine(conn)

bl.print_range('05RG101', '05RG102')
#print "-------------"
#bl.print_range('06RG101', '06RG102')
print " ########################"
#hp.set_db_epics('MRPH001D01', '100.0', conn, bl)
hp.set_db_epics('05KS001E01', '-20.0', conn, bl)
bl.print_range('05RG101', '05RG102')
#print "-------------"
#bl.print_range('06RG101', '06RG102')
print " ########################"
#hp.set_db_epics('MRPH001D01', '70.0', conn, bl)
hp.set_db_epics('05KS001E01', '-36.0', conn, bl)
bl.print_range('05RG101', '05RG102')
#print "-------------"
#bl.print_range('06RG101', '06RG102')
