#SqlDb.py
"""Python access to the linac databases for quicker read access
to various quantities, e.g. element lenght"""

import sqlite3 as lite
import sys
import os

class Db_bl():
    """An class for accessing the given beamline databases through SQLite"""

# functions 
    def __init__(self, db_dir, db_list):
        """Creates database beamline object that contains dir and names
        of beamline db files
        db_dir,    // path to db directory
        db_list    // list containing names of databases"""
        self.db_dir = db_dir
        self.db_list = db_list

    def get_bl_elem_len(self):
        """Returns a sorted list of lists containing the
        view_index, element_name, element_length, cumulative_length"""
        # use this list to check for db elements that are part of the beamline
        bl_elem_list = ['buncher', 'caperture', 'diagnostics', 'dipole', 
                        'drift', 'quad', 'raperture', 'rf_gap', 'rotation', 
                        'spch_comp']
        output = []
        for item in self.db_list:
            #db connection
            #self.db_dir + '/' + item
            dbf_path = os.path.join(os.path.abspath(self.db_dir), item)
            # READ ONLY access to db in python 2.X
            fd = os.open(dbf_path, os.O_RDONLY)
            con = lite.connect('/dev/fd/%d' % fd)
            cur = con.cursor()
            # get db table names 
            db_query = "SELECT name  FROM sqlite_master  WHERE type='table' \
                       ORDER by NAME"
            cur.execute(db_query)
            results = [] #view index, name, elem_length (m)
            all_tables = map(lambda t: t[0], cur.fetchall())            
            for tblname in all_tables:
                # get column names each table in list
                if tblname in bl_elem_list:
                    tbl_query = "SELECT * FROM %s " % tblname
                    col_names = list(map(lambda x: x[0], 
                                     cur.execute(tbl_query).description))
                    # if view_index present than this is part of actual beamline 
                    if 'view_index' in col_names:
                        if tblname == 'dipole':
                            # must calc effective path length from rho and theta
                            tbl_query = "SELECT view_index, name, rho_model, \
                                         angle_model FROM %s ORDER BY \
                                         view_index ASC" % (tblname)
                            cur.execute(tbl_query)
                            temp_table = cur.fetchall()
                            temp_results = []
                            for item in temp_table: 
                                temp_results.append([item[0]] + 
                                                    [item[1]] + 
                                                    [item[2] * item[3]]) 
                            results += temp_results

                        else:
                            # other tables that may store length directly
                            # get name of actual length field
                            len_name = filter((lambda x: x.find('length') > -1), 
                                              col_names)
                            if len(len_name) > 0:
                                item = len_name[0]
                                # read view index, name and length from table
                                tbl_query = "SELECT view_index, name, %s FROM \
                                             %s ORDER BY view_index ASC" % (item, 
                                                                         tblname)
                                cur.execute(tbl_query)
                                temp_table = cur.fetchall()
                                table = list(map(lambda x: list(x), temp_table))
                                results += table

                            else:
                                # print 'zero length elements in table', tblname
                                # read view index and name and add zero length 
                                tbl_query = "SELECT view_index, name FROM %s \
                                             ORDER BY view_index ASC" % (tblname)
                                cur.execute(tbl_query)
                                temp_table = cur.fetchall()
                                table = []
                                for i in range(len(temp_table)):
                                    table.append(list(temp_table[i]) + [0.0])

                                results += table

            output += sorted(results, key = (lambda x: x[0]))
            #con.close()
            os.close(fd)

        #add to list the cumulative length to end of each element
        ltot = 0.0
        for elem in output:
            ltot += elem[2]
            # convert db names from unicode to ascii format
            elem[1]=elem[1].encode('ascii','ignore')
            elem.append(ltot)
        #access by index from 0 to len-1, same scheme used by HPSim
        return output
