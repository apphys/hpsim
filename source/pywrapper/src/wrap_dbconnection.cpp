#include <iostream>
#include "wrap_dbconnection.h"
#include "cppclass_object.h"
#include "sql_utility.h"

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* DBConnectionNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(dbconnection_init__doc__,
"DBConnection(db_address)\n\n"
"DBConnection class."
);
static int DBConnectionInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  char* db_addr;
  if(!PyArg_ParseTuple(args, "s:__init__", &db_addr))
  {
    std::cerr << "DBConnection constructor needs a db address!"<< std::endl;
    return 0;
  }
  self->cpp_obj = new DBConnection(std::string(db_addr));
  ((DBConnection*) self->cpp_obj)->SetWrapper((PyObject*) self);
  return 0;
}

static void DBConnectionDel(CPPClassObject* self)
{
  delete (DBConnection*)(self->cpp_obj);
  self->ob_type->tp_free((PyObject*) self);
}

PyDoc_STRVAR(load_lib__doc__,
"load_lib(lib_address) ->\n\n"
"Load an external library."
);
static PyObject* DBConnectionLoadLib(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  DBConnection* dbconn = (DBConnection*)(cppclass_obj->cpp_obj); 
  char* lib;
  if(!PyArg_ParseTuple(args, "s", &lib))
  {
    std::cerr << "DBConnection.load_lib needs a lib address!"<< std::endl;
    return NULL;
  }
  dbconn->LoadLib(std::string(lib));
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(attach_db__doc__,
"attach_db(db_address, db_name) -> \n\n"
"Attach a db."
);
static PyObject* DBConnectionAttachDB(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  DBConnection* dbconn = (DBConnection*)(cppclass_obj->cpp_obj); 
  char* db_addr, *db_name;
  if(!PyArg_ParseTuple(args, "s|s", &db_addr, &db_name))
  {
    std::cerr << "DBConnection.attach_db needs a db address and a name(optional)!"<< std::endl;
    return NULL;
  }
  dbconn->AttachDB(std::string(db_addr), std::string(db_name));
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(print_dbs__doc__,
"print_dbs() ->\n\n"
"Print all the attached dbs."
);
static PyObject* DBConnectionPrintDBs(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  DBConnection* dbconn = (DBConnection*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  dbconn->PrintDBs();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(print_libs__doc__,
"print_libs() -> \n\n"
"Print all the external libraries."
);
static PyObject* DBConnectionPrintLibs(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  DBConnection* dbconn = (DBConnection*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  dbconn->PrintLibs();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(clear_model_index__doc__,
"clear_model_index() -> \n\n"
"Clear model indices in the dbs."
);
static PyObject* DBConnectionClearModelIndex(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  DBConnection* dbconn = (DBConnection*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  dbconn->ClearModelIndex();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_epics_channels__doc__,
"get_epics_channels() -> list(string)\n\n"
"Get a list of EPICS channels in the dbs."
);
static PyObject* DBConnectionGetEPICSChannels(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  DBConnection* dbconn = (DBConnection*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  std::vector<std::string> pvs = dbconn->GetEPICSChannels();
  if(pvs.empty())
    return NULL;
  PyObject* pv_lst = PyList_New(pvs.size());
  for(int i = 0; i < pvs.size(); ++i)
    PyList_SetItem(pv_lst, i, PyString_FromString(pvs[i].c_str())); 
  return pv_lst;
}
static PyMethodDef DBConnectionMethods[] = {
  {"load_lib", DBConnectionLoadLib, METH_VARARGS, load_lib__doc__},
  {"attach_db", DBConnectionAttachDB, METH_VARARGS, attach_db__doc__},
  {"print_dbs", DBConnectionPrintDBs, METH_VARARGS, print_dbs__doc__},
  {"print_libs", DBConnectionPrintLibs, METH_VARARGS, print_libs__doc__},
  {"clear_model_index", DBConnectionClearModelIndex, METH_VARARGS, clear_model_index__doc__},
  {"get_epics_channels", DBConnectionGetEPICSChannels, METH_VARARGS, get_epics_channels__doc__},
  {NULL}
};

static PyMemberDef DBConnectionMembers[] = {
  {NULL}
};

static PyTypeObject DBConnection_Type = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "DBConnection", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) DBConnectionDel, /*tp_dealloc*/
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
    dbconnection_init__doc__,  /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    DBConnectionMethods, /* tp_methods */
    DBConnectionMembers, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) DBConnectionInit, /* tp_init */
    0, /* tp_alloc */
    DBConnectionNew, /* tp_new */
};

PyMODINIT_FUNC initDBConnection(PyObject* module)
{
  if(PyType_Ready(&DBConnection_Type) < 0) return;
  Py_INCREF(&DBConnection_Type);
  PyModule_AddObject(module, "DBConnection", (PyObject*)&DBConnection_Type);
}

#ifdef _cplusplus
}
#endif
