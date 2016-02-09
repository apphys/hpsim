#ifndef INIT_PV_OBSERVER_LIST_H
#define INIT_PV_OBSERVER_LIST_H
#include "beamline.h"
#include "pv_observer_list.h"
#include "sql_utility.h"

void InitPVObserverList(PVObserverList&, BeamLine&, DBConnection&);

#endif
