#ifndef INIT_PVLIST_H
#define INIT_PVLIST_H
#include "beamline.h"
#include "pvlist.h"
#include "sql_utility.h"

void InitPVList(PVList&, BeamLine&, DBConnection&);

#endif
