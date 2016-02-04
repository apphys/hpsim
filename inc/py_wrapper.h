#ifndef PY_WRAPPER_H
#define PY_WRAPPER_H

#include <Python.h>

class PyWrapper
{
public:
  PyWrapper(PyObject*);
  PyWrapper();
  ~PyWrapper(){}
  void SetWrapper(PyObject* r_wrapper)
  {
    py_wrapper_ = r_wrapper;
  }
  PyObject* GetWrapper()
  {
    return py_wrapper_;
  }
private:
  PyObject* py_wrapper_;
};
#endif
