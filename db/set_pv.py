import sqlite3 as lite
from subprocess import call
import os

dbfiles = ['hb16-tbtd.db', 'hb16-dtl.db']

for df in dbfiles:
  con = lite.connect(df)
  with con:
    cur = con.cursor()
    cur.execute("select lcs_name, value from epics_channel")
    values = cur.fetchall()
    for v in values:
      print '--caput ' + str(v[0]) + ' ' + str(v[1])
      os.system('caput ' + str(v[0]) + ' ' + str(v[1]))
  con.close()
