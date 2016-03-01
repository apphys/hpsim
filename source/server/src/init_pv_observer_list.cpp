#include <iostream>
#include "init_pv_observer_list.h"


void InitPVObserverList(PVObserverList& r_pvlist, BeamLine& r_bl, DBConnection& r_dbcon, bool r_verbose)
{
  std::vector<std::string> dbs = r_dbcon.dbs;
  sqlite3* db_conn = r_dbcon.db_conn;

  for(int dbs_indx = 0; dbs_indx < dbs.size(); ++dbs_indx)
  {
    std::string db = dbs[dbs_indx]; 
    std::string sql = "select lcs_name, value_type from " + db + ".epics_channel";
    std::vector<std::vector<std::string> > pv_list = GetQueryResults(db_conn, sql.c_str());

    for(int i = 0; i < pv_list.size(); ++i)
    {
      std::string pv = pv_list[i][0]; 
      std::string pv_type = pv_list[i][1];
      sql = "select model_type, name from " + db + ".channel_list where channel1='" + 
        pv + "' or channel2='" + pv + "' or channel3 ='" + pv + "'";
      std::vector<std::vector<std::string> > elems_info = GetQueryResults(db_conn, sql.c_str());
      for(int j = 0; j < elems_info.size(); ++j)
      {
        std::string elem_type = elems_info[j][0];
        std::string elem_name = elems_info[j][1];
        // first time, create the PVObserver
        if(r_pvlist[pv] == NULL) 
        { 
          if(elem_type == "quad")
            r_pvlist.AddPVObserver(pv, new QuadPVObserver(pv, db));    
          else if(elem_type == "rf_module")
          {
            if(pv_type == "rf_ph")
              r_pvlist.AddPVObserver(pv, new RFPhasePVObserver(pv, db));
            else if(pv_type == "rf_amp" || pv_type == "delay") 
              r_pvlist.AddPVObserver(pv, new RFAmplitudePVObserver(pv, db));
          }
          else if(elem_type == "buncher")
          {
            if(pv_type == "buncher_ph") 
              r_pvlist.AddPVObserver(pv, new BuncherPhasePVObserver(pv, db));
            else if(pv_type == "buncher_amp")
              r_pvlist.AddPVObserver(pv, new BuncherAmplitudePVObserver(pv, db));
            else if(pv_type == "buncher_on_off") 
              r_pvlist.AddPVObserver(pv, new BuncherOnOffPVObserver(pv, db));
          }
          else if(elem_type == "dipole")
            r_pvlist.AddPVObserver(pv, new DipolePVObserver(pv, db));
	  if(r_verbose)
	    std::cout << "---------- Add PV: " << pv << std::endl;
        }
        // Attach BeamLineElement to the PVObserver
        if(elem_type != "rf_module" && r_bl[elem_name] != NULL)
        {
            r_pvlist.AttachBeamLineElementToPVObserver(pv, r_bl[elem_name]);
	    if(r_verbose)
	      std::cout << "Attach " << elem_name << " to " << pv << std::endl;
        }
        else if (elem_type == "rf_module")
        {
          sql = "select g.name from " + db + ".rf_gap g join rf_module m on m.id = g.module_id where m.name = '" + elem_name + "'";  
          std::vector<std::vector<std::string> > gap_names = GetQueryResults(db_conn, sql.c_str());
          for(int gp = 0; gp < gap_names.size(); ++gp)
          {
            r_pvlist.AttachBeamLineElementToPVObserver(pv, r_bl[gap_names[gp][0]]);
	    if(r_verbose)
	      std::cout << "Attach " << gap_names[gp][0] << " to " << pv << std::endl;
          }
        }
        else
          std::cerr << "InitPVObserverList error: Cannot find " << elem_name << " in beamline!" << std::endl;
      }// for j elems_info
    }// for i pv_list
  }// for dbs_indx
  r_pvlist.SetDBconn(db_conn);
}
