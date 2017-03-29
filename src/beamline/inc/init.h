#ifndef INIT_H
#define INIT_H

#include "beamline.h"
#include "sql_utility.h"

void SetGPU(int);
void GenerateBeamLine(BeamLine& r_linac, DBConnection*);

#endif
