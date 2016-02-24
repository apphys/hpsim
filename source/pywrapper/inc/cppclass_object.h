#ifndef CPPCLASS_OBJECT_H
#define CPPCLASS_OBJECT_H

#include <structmember.h>

#ifdef _cplusplus
extern "C" {
#endif

typedef struct{
  PyObject_HEAD
  void* cpp_obj;
} CPPClassObject;

#ifdef _cplusplus
}
#endif
#endif
