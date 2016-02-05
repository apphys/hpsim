#include <utility>
#include <iterator>
#include "pvlist.h"
#include "pthread_check.h"

PVList::~PVList()
{
  std::map<std::string, Updater*>::iterator iter;
  for (iter = list.begin(); iter != list.end(); ++iter) 
    delete iter->second;
}

void PVList::AddChannel(std::string r_pv, std::string r_db)
{
  Updater* tmp = new Updater(r_pv);
  tmp->SetDB(r_db);
  list[r_pv] = tmp;
}

void PVList::SetDBconn(sqlite3* r_db)
{
  std::map<std::string, Updater*>::iterator iter;
  for (iter = list.begin(); iter != list.end(); ++iter) 
  {     
    (iter->second)->SetDBconn(r_db);
  }
}

void PVList::SetBeamLine(BeamLine* r_bl)
{
  std::map<std::string, Updater*>::iterator iter;
  for (iter = list.begin(); iter != list.end(); ++iter) 
  {     
    (iter->second)->SetBeamLine(r_bl);
  }
}
