#include "wrap_spacecharge.h"
#include "space_charge.h"
#include "hpsim_module.h"
#include "cppclass_object.h"
#include <iostream>
#include <string>

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* SpaceChargeNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(spacecharge_init__doc__,
"SpaceCharge(nr = grid_number_r, nz = grid_number_z, "
"interval = max_drift_length_without_kick (optional), "
"adj_bunch = number_adjacent_bunches (optional), "
"type = routine_name (optional))\n"
"nr: number of grids in r direction.\n"
"nz: number of grids in z direction.\n"
"interval: the maximum drift length without a space charge kick, default = 0.01m.\n"
"adj_bunch: number of adjacent bunches used in low energy transport, default 0.\n"
"type: \"scheff\" (default)\n\n"
"The SpaceCharge class."
);
static int SpaceChargeInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  int nr = 32, nz = 64, adj_bunch = 0;
  double interval = 0.01;
  char* type = "scheff";
  static char *kwlist[] = {"nr", "nz", "interval", "adj_bunch", "type", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii|dis:__init__", kwlist, 
                      &nr, &nz, &interval, &adj_bunch, &type))
  {
    std::cerr << "SpaceCharge needs at least nr and nz as its input." 
              << std::endl;
    return 0;
  }
  if(std::string(type) == "scheff")
  {
    self->cpp_obj = new Scheff(nr, nz, adj_bunch);
    ((SpaceCharge*) self->cpp_obj)->SetInterval(interval);
    ((SpaceCharge*) self->cpp_obj)->SetWrapper((PyObject*) self);
  }
  else
    std::cerr << "SpaceCharge type: " << type << " is not supported." << std::endl;

  return 0;
}

static void SpaceChargeDel(CPPClassObject* self)
{
  delete (SpaceCharge*)(self->cpp_obj);
  self->ob_type->tp_free((PyObject*) self);
}

PyDoc_STRVAR(set_mesh_size__doc__,
"set_mesh_size(nr = grid_number_r, nz = grid_number_z)->\n"
"nr: number of grids in r direction.\n"
"nz: number of grids in z direction.\n\n"
"Set the size of the mesh."
);
static PyObject* SetMeshSize(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  double nr, nz;
  static char* kwlist[] = {"nr", "nz", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "dd:set_mesh_size", kwlist, &nr, &nz))
  {
    std::cerr << "set_mesh_size() needs nr and nz as its input." << std::endl;
    return NULL;
  }
  spch->SetMeshSize(nr, nz);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_mesh_size__doc__,
"get_mesh_size() -> list(float, float)\n\n"
"Get the 2D mesh size."
);
static PyObject* GetMeshSize(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;

  double nr = spch->GetNx();
  double nz = spch->GetNz();
  PyObject* lst = PyList_New(2);
  PyList_SET_ITEM(lst, 0, PyFloat_FromDouble(nr)) ;
  PyList_SET_ITEM(lst, 1, PyFloat_FromDouble(nz)) ;
  return lst; 
}

PyDoc_STRVAR(set_interval__doc__,
"set_interval(max_drift_length_without_kick)->\n\n"
"Set the the maximum drift length without a space charge kick."
);
static PyObject* SetInterval(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  double inter;
  if(!PyArg_ParseTuple(args, "d:set_interval", &inter))
    return NULL;
  spch->SetInterval(inter);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_interval__doc__,
"get_interval() -> float \n\n"
"Get the maximum drift length without a space charge kick."
);
static PyObject* GetInterval(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  return PyFloat_FromDouble(spch->GetInterval());
}

PyDoc_STRVAR(set_adj_bunch__doc__,
"set_adj_bunch(number_adjacent_bunch)->\n\n"
"Set the number of adjacent bunches used in low energy transport."
);
static PyObject* SetAdjBunch(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  double adjb;
  if(!PyArg_ParseTuple(args, "d:set_adj_bunch", &adjb))
    return NULL;
  spch->SetAdjBunch(adjb);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_adj_bunch__doc__,
"get_adj_bunch() -> float \n\n"
"Get the number of adjacent bunches used in low energy transport."
);
static PyObject* GetAdjBunch(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  return PyFloat_FromDouble(spch->GetAdjBunch());
}

PyDoc_STRVAR(set_adj_bunch_cutoff_w__doc__,
"set_adj_bunch_cutoff_w(energy)->\n\n"
"Set the cutoff energy of the beam at which the simulation swiches to not using adjacent bunchs and calculate the mash region using 3*sigmas. "
);
static PyObject* SetAdjBunchCutoffW(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  double adjb;
  if(!PyArg_ParseTuple(args, "d:set_adj_bunch_cutoff_w", &adjb))
    return NULL;
  spch->SetAdjBunchCutoffW(adjb);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_adj_bunch_cutoff_w__doc__,
"get_adj_bunch_cutoff_w() -> float\n\n"
"Get the cutoff energy of the beam at which the simulation swiches to not using adjacent bunchs and calculate the mash region using 3*sigmas."
);
static PyObject* GetAdjBunchCutoffW(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  return PyFloat_FromDouble(spch->GetAdjBunchCutoffW());
}

PyDoc_STRVAR(set_mesh_size_cutoff_w__doc__,
"set_mesh_size_cutoff_w(energy) -> \n\n"
"Set the cutoff energy of the beam at which the mesh sizes will change nr = nr/2, nz = nz/2, interval = interval * 4."
);
static PyObject* SetMeshSizeCutoffW(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  double adjb;
  if(!PyArg_ParseTuple(args, "d:set_mesh_size_cutoff_w", &adjb))
    return NULL;
  spch->SetMeshSizeCutoffW(adjb);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_mesh_size_cutoff_w__doc__,
"get_mesh_size_cutoff_w() -> float\n\n"
"Get the cutoff energy of the beam at which the mesh sizes will change nr = nr/2, nz = nz/2, interval = interval * 4."
);
static PyObject* GetMeshSizeCutoffW(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  return PyFloat_FromDouble(spch->GetMeshSizeCutoffW());
}

PyDoc_STRVAR(get_remesh_threshold__doc__,
"get_remesh_threshold() -> float\n\n"
"Get the remesh threshold value, default = 0.05."
);
static PyObject* GetRemeshThreshold(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj);
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  return PyFloat_FromDouble(spch->GetRemeshThreshold());
}

PyDoc_STRVAR(set_remesh_threshold__doc__,
"set_remesh_threshold(value) -> \n\n"
"Set the remesh threshold value, default = 0.05."
);
static PyObject* SetRemeshThreshold(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SpaceCharge* spch = (SpaceCharge*)(cppclass_obj->cpp_obj);
  double thr;
  if(!PyArg_ParseTuple(args, "d:set_remesh_threshold", &thr))
    return NULL;
  spch->SetRemeshThreshold(thr);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef SpaceChargeMethods[] = {
  {"set_mesh_size", (PyCFunction)SetMeshSize, METH_VARARGS|METH_KEYWORDS, set_mesh_size__doc__},
  {"get_mesh_size", GetMeshSize, METH_VARARGS, get_mesh_size__doc__},
  {"get_interval", GetInterval, METH_VARARGS, get_interval__doc__},
  {"set_interval", SetInterval, METH_VARARGS, set_interval__doc__},
  {"get_adj_bunch", GetAdjBunch, METH_VARARGS, get_adj_bunch__doc__},
  {"set_adj_bunch", SetAdjBunch, METH_VARARGS, set_adj_bunch__doc__},
  {"get_adj_bunch_cutoff_w", GetAdjBunchCutoffW, METH_VARARGS, get_adj_bunch_cutoff_w__doc__},
  {"set_adj_bunch_cutoff_w", SetAdjBunchCutoffW, METH_VARARGS, set_adj_bunch_cutoff_w__doc__},
  {"get_mesh_size_cutoff_w", GetMeshSizeCutoffW, METH_VARARGS, get_mesh_size_cutoff_w__doc__},
  {"set_mesh_size_cutoff_w", SetMeshSizeCutoffW, METH_VARARGS, set_mesh_size_cutoff_w__doc__},
  {"get_remesh_threshold", GetRemeshThreshold, METH_VARARGS, get_remesh_threshold__doc__},
  {"set_remesh_threshold", SetRemeshThreshold, METH_VARARGS, set_remesh_threshold__doc__},
  {NULL, NULL}
};

static PyMemberDef SpaceChargeMembers[] = {
  {NULL}
};

static PyTypeObject SpaceCharge_Type = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "SpaceCharge", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) SpaceChargeDel, /*tp_dealloc*/
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    spacecharge_init__doc__, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    SpaceChargeMethods, /* tp_methods */
    SpaceChargeMembers, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) SpaceChargeInit, /* tp_init */
    0, /* tp_alloc */
    SpaceChargeNew, /* tp_new */
};

PyMODINIT_FUNC initSpaceCharge(PyObject* module)
{
  if(PyType_Ready(&SpaceCharge_Type) < 0) return;
  Py_INCREF(&SpaceCharge_Type);
  PyModule_AddObject(module, "SpaceCharge", (PyObject*)&SpaceCharge_Type);
}


#ifdef _cplusplus
}
#endif


