#ifndef HPSIM_MODULE_H
#define HPSIM_MODULE_H

#ifdef _cplusplus
extern "C" {
#endif
PyMODINIT_FUNC initHPSim();
PyObject* getHPSimType(char* name);
#ifdef _cplusplus
}
#endif
#endif

