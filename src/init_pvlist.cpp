#include "init_pvlist.h"


void InitPVList(PVList& r_pvlist, BeamLine& r_bl, DBConnection& r_dbcon)
{
  std::vector<std::string> dbs = r_dbcon.dbs;
  sqlite3* db_conn = r_dbcon.db_conn;

  for(int dbs_indx = 0; dbs_indx < dbs.size(); ++dbs_indx)
  {
    std::string sql = "select lcs_name from " + dbs[dbs_indx] + ".epics_channel";
    std::vector<std::string> pv_list = GetDataArrayFromDB(db_conn, sql.c_str());
    for(int i = 0; i < pv_list.size(); ++i)
      r_pvlist.AddChannel(pv_list[i], dbs[dbs_indx]);
  }
  r_pvlist.SetDBconn(db_conn);
  r_pvlist.SetBeamLine(&r_bl);
}
