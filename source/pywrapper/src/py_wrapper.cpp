#include "py_wrapper.h"

PyWrapper::PyWrapper(PyObject* r_wrapper)
  : py_wrapper_(r_wrapper)
{
}

PyWrapper::PyWrapper() : py_wrapper_(NULL)
{
}
