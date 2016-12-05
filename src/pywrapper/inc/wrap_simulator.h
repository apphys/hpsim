#ifndef WRAP_SIMULATOR_H
#define WRAP_SIMULATOR_H
#include <Python.h>

#ifdef _cplusplus
extern "C" {
#endif
PyMODINIT_FUNC initSimulator(PyObject* module);
#ifdef _cplusplus
}
#endif
#endif
