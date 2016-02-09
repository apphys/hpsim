#ifndef PV_OBSERVER_LIST_H
#define PV_OBSERVER_LIST_H

#include <map>
#include "pv_observer.h"
#include "beamline_element.h"

class PVObserverList
{
public:
  typedef std::map<std::string, PVObserver*>::iterator MapIter;
  ~PVObserverList();
  void AddPVObserver(std::string, PVObserver*);
  void AttachBeamLineElementToPVObserver(std::string, BeamLineElement*);
  size_t GetSize() const;
  void SetDBconn(sqlite3*);
  PVObserver* operator[](std::string r_pv);
  void Print();
  std::map<std::string, PVObserver*>& GetList();
private:
  std::map<std::string, PVObserver*> list_;
};

inline 
std::map<std::string, PVObserver*>& PVObserverList::GetList()
{
  return list_;
}
inline
size_t PVObserverList::GetSize() const
{
  return list_.size();
}

inline
PVObserver* PVObserverList::operator[](std::string r_pv)
{
  if(list_.find(r_pv) == list_.end())
    return NULL;
  return list_[r_pv];
}

#endif
