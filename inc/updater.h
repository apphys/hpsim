#ifndef UPDATER_H
#define UPDATER_H
#include <string>
#include <vector>
#include <sqlite3.h>
#include "beamline.h"
#include <stdlib.h>
#include <cadef.h>

class Updater
{
public:
  Updater(std::string r_pv);  
  std::string GetPV() const
  {
    return pv_;
  }
  void SetPV(std::string r_pv)
  {
    pv_ = r_pv;
  }
  void SetDBconn(sqlite3* r_db_conn)
  {
    db_conn_ = r_db_conn;
  }
  void SetDB(std::string r_db)
  {
    db_ = r_db;
  }
  void SetBeamLine(BeamLine* r_bl)
  {
    beamline_ptr_ = r_bl;
  }
  bool NeedUpdate()
  {
    if (old_val_ == "")
      return true;
    double v = atof(val_.c_str());
    double ov = atof(old_val_.c_str());
    if(v - ov < 1e-6 || ov - v < 1e-6)
      return false;
    return true;
  }
  void UpdateOldValue() 
  {
    old_val_ = val_;
  }
  void UpdateDB();
  void UpdateModel();
  void SetValue(std::string r_val)
  {
    val_ = r_val;
  }
//private:
  std::string pv_;
  std::string val_; 
  std::string old_val_;
  sqlite3* db_conn_;
  std::string db_;
  /* Pointer to the start of the beam line*/
  BeamLine* beamline_ptr_; 
};

//class EPICSUpdater : public Updater
//{
//public:
//  EPICSUpdater(std::string r_pv, bool r_istxt = false);
////  ~EPICSUpdater();
//  void Init();
//  bool ReadFromEPICS();
//  bool istxt;
//private:
////  pv* epics_pv_;
//  void connection_handler(struct connection_handler_args args); 
//  void event_handler (evargs args);
//};
#endif
