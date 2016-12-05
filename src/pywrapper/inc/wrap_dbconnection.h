#ifndef WRAP_DBCONNECTION_H
#define WRAP_DBCONNECTION_H

#include <Python.h>

#ifdef _cplusplus
extern "C" {
#endif

PyMODINIT_FUNC initDBConnection(PyObject* module);

#ifdef _cplusplus
}
#endif
#endif
