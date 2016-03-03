#include <iostream>
#include "wrap_beamline.h"
#include "hpsim_module.h"
#include "cppclass_object.h"
#include "beamline.h"
#include "init.h"
#include "sql_utility.h"

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* BeamLineNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(beamline_init__doc__, 
"BeamLine(DBConnection)\n\n"
"The BeamLine class."
);
static int BeamLineInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
  PyObject* py_dbconnection;
  if(!PyArg_ParseTuple(args, "O:__init__", &py_dbconnection))
  {
    std::cerr << "BeamLine constructor needs a DBConnection!"<< std::endl;
    return 0;
  }
  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
  {
    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
    self->cpp_obj = new BeamLine();
    GenerateBeamLine(*((BeamLine*)self->cpp_obj), dbconn);

    ((BeamLine*) self->cpp_obj)->SetWrapper((PyObject*) self);
  }
  return 0;
}

static void BeamLineDel(CPPClassObject* self)
{
  delete (BeamLine*)(self->cpp_obj);
  self->ob_type->tp_free((PyObject*) self);
}

PyDoc_STRVAR(print_out__doc__, 
"print_out()->\n\n"
"Print the beamline settings to terminal."
);
static PyObject* BeamLinePrint(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  bl->Print();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(print_range__doc__,
"print_range(start_element, end_element)->\n\n"
"Print the beamline in the range of [start, end] (inclusive)."
);
static PyObject* BeamLinePrintRange(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem, *end_elem;
  if(!PyArg_ParseTuple(args, "ss", &start_elem, &end_elem))
  {
    std::cerr << "BeamLine.print_range needs two elem names as args!" << std::endl;
    return NULL;
  }
  bl->Print(start_elem, end_elem);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_element_names__doc__,
"get_element_names(start=start_element(optional), end=end_element(optional), type=type(optional))->\n\n"
"Return a list of beamline element names in the range of [start, end] (inclusive). \n"
"If no start_element is specified, it will start from the beginning of the beamline. \n"
"If no end_element is specified, it will end at the last element of the beamline. \n"
"type can be 'ApertureC' 'ApertureR', 'Buncher', 'Diagnostics', 'Dipole', 'Drift', 'Quad',\n"
" 'RFGap-DTL', 'RFGap-CCL', 'Rotation', 'SpchComp' "
);
static PyObject* BeamLineGetElementNames(PyObject* self, PyObject* args, PyObject* kwds)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  BeamLine* bl = (BeamLine*)(cppclass_obj->cpp_obj); 
  char* start_elem = "", *end_elem = "",  *type = "";
  static char *kwlist[] = {"start", "end", "type", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sss", kwlist, &start_elem, &end_elem, &type))
    return NULL;
  std::vector<std::string> names = bl->GetElementNames(start_elem, end_elem,type);
  if(!names.empty())
  {
    PyObject* elem_lst = PyList_New(names.size());
    for(int i = 0; i < names.size(); ++i)
    PyList_SetItem(elem_lst, i, PyString_FromString(names[i].c_str()));
    return elem_lst;
  }
  Py_INCREF(Py_None);
  return Py_None;
}
static PyMethodDef BeamLineMethods[] = {
  {"print_out", BeamLinePrint, METH_VARARGS, print_out__doc__},
  {"print_range", BeamLinePrintRange, METH_VARARGS, print_range__doc__},
  {"get_element_names", (PyCFunction)BeamLineGetElementNames, METH_VARARGS|METH_KEYWORDS, get_element_names__doc__},
  {NULL}
};

static PyMemberDef BeamLineMembers[] = {
  {NULL}
};

static PyTypeObject BeamLine_Type = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "BeamLine", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) BeamLineDel, /*tp_dealloc*/
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
    beamline_init__doc__, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    BeamLineMethods, /* tp_methods */
    BeamLineMembers, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) BeamLineInit, /* tp_init */
    0, /* tp_alloc */
    BeamLineNew, /* tp_new */
};

PyMODINIT_FUNC initBeamLine(PyObject* module)
{
  if(PyType_Ready(&BeamLine_Type) < 0) return;
  Py_INCREF(&BeamLine_Type);
  PyModule_AddObject(module, "BeamLine", (PyObject*)&BeamLine_Type);
}


#ifdef _cplusplus
}
#endif


