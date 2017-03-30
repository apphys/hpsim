#include <iostream>
#include <algorithm>
#include <iterator>
#include "pv_observer_list.h"

PVObserverList::~PVObserverList()
{
  for(MapIter iter = list_.begin(); iter != list_.end(); ++iter)
    if(iter->second != NULL)
      delete iter->second;
}

void PVObserverList::SetDBconn(sqlite3* r_db_conn)
{
  for(MapIter iter = list_.begin(); iter != list_.end(); ++iter)
    if(iter->second != NULL)
      iter->second->SetDBconn(r_db_conn); 
}

void PVObserverList::AddPVObserver(std::string r_pv, PVObserver* r_ob)
{
  if(list_.find(r_pv) != list_.end())
    std::cout << "PVObserver for " << r_pv << "exists, overwrite! " << std::endl;
  list_[r_pv] = r_ob;
}

void PVObserverList::AttachBeamLineElementToPVObserver(std::string r_pv, 
  BeamLineElement* r_elem)
{
  if(list_.find(r_pv) == list_.end())
  {
    std::cerr << "PVObserver for " << r_pv << 
      " does not exist, cannot attach element!" << std::endl;
    exit(-1);
  }
  list_[r_pv]->AttachBeamLineElement(r_elem);
}

void PVObserverList::Print()
{
  for(MapIter iter = list_.begin(); iter != list_.end(); ++iter)
  {
    std::cout << iter->first << ": ";
    if(iter->second != NULL)
    {
      std::vector<std::string> elems =  iter->second->GetBeamLineElementNames();
      std::copy(elems.begin(), elems.end(), std::ostream_iterator<std::string>(
	std::cout, " "));
    }
    std::cout << std::endl;
  }
}
