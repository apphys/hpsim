#include <Python.h>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include "pyepics_module.h"
#include "cppclass_object.h"
#include "cadef.h"
#include "ca_check.h"

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* CAGet(PyObject* self, PyObject* args)
{
  char* pv_name;
  if(!PyArg_ParseTuple(args, "s", &pv_name))
  {
    std::cerr << "caget only takes a pv name as its arg!" << std::endl;
    return NULL;
  }

  chid mychid;
  double val_dbl = 0.0;
  if(!CACheck(ca_create_channel(pv_name, NULL,NULL,10,&mychid), "ca_create_channel", pv_name))
    return false;
  if(!CACheck(ca_pend_io(5.0), "ca_pend_io after ca_create_channel", pv_name))
    return false;
  if(!CACheck(ca_get(DBR_DOUBLE, mychid, (void*)&val_dbl), "ca_get", pv_name))
    return false;
  if(!CACheck(ca_pend_io(5.0), "ca_pend_io after ca_get", pv_name))
    return false;
  if(!CACheck(ca_clear_channel(mychid), "ca_clear_channel", pv_name))
    return false;

  return PyFloat_FromDouble(val_dbl);
}

static PyMethodDef PyEPICSModuleMethods[]={
  {"caget", (PyCFunction)CAGet, METH_VARARGS, "python ca_get"}, 
  {NULL}
};

PyMODINIT_FUNC initPyEPICS()
{
  PyObject* module = Py_InitModule("PyEPICS", PyEPICSModuleMethods);
}

/*
PyObject* getPyEPICSType(char* name)
{
  PyObject* mod = PyImport_ImportModule("PyEPICS");
  PyObject* pyType = PyObject_GetAttrString(mod, name);
  Py_DECREF(mod);
  Py_DECREF(pyType);
  return pyType;
}
*/
#ifdef _cplusplus
}
#endif
