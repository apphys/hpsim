#ifndef PVLIST_H
#define PVLIST_H
#include <map>
#include <string>
#include <pthread.h>
#include <sqlite3.h>
#include "beamline_element.h"
#include "updater.h"

struct PVList
{
  PVList(){}
  ~PVList();
  void AddChannel(std::string r_pv, std::string r_db);
  void SetDBconn(sqlite3* r_db);
  void SetBeamLine(BeamLine* r_bl);
  int GetListSize() const
  {
    return list.size();
  }
  Updater* operator[](std::string r_pv)
  {
      return list[r_pv];
  }
  std::map<std::string, Updater*> list;
};
#endif
