#include "wrap_simulator.h"
#include "simulation_engine.h"
#include "hpsim_module.h"
#include "cppclass_object.h"
#include <iostream>
#include <algorithm>
#include <string>

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* SimulatorNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(simulator_init__doc__,
"Simulator(beam = Beam, beamline = BeamLine, spch = SpaceChargeRoutine (optional))\n"
"beam: a Beam object\n"
"beamline: a BeamLine object\n"
"spch (optional): a SpaceCharge object, if not defined, then no space charge is applied in the simulation.\n\n"
"The Simulator class."
);
static int SimulatorInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  PyObject* py_beam, *py_beamline, *py_spch = NULL;
  static char *kwlist[] = {"beam", "beamline", "spch", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:__init__", kwlist, 
                      &py_beam, &py_beamline, &py_spch))
  {
    std::cerr << "Simulator constructor needs at least a beam and beamline pointer as its args!"
              << std::endl;
    return 0;
  }
  PyObject* py_beam_type = getHPSimType("Beam");
  PyObject* py_beamline_type = getHPSimType("BeamLine");
  PyObject* py_spacecharge_type = getHPSimType("SpaceCharge");
  if(PyObject_IsInstance(py_beam, py_beam_type) && 
     PyObject_IsInstance(py_beamline, py_beamline_type))
  {
    Beam* beam = (Beam*)((CPPClassObject*)py_beam)->cpp_obj;
    BeamLine* beamline = (BeamLine*)((CPPClassObject*)py_beamline)->cpp_obj;
    SpaceCharge* scheff = NULL;
    if(py_spch != NULL && PyObject_IsInstance(py_spch, py_spacecharge_type)) 
      scheff = (SpaceCharge*)((CPPClassObject*)py_spch)->cpp_obj;
    self->cpp_obj = new SimulationEngine();
    ((SimulationEngine*) self->cpp_obj)->SetWrapper((PyObject*) self);
    ((SimulationEngine*) self->cpp_obj)->InitEngine(beam, beamline, scheff);
  }
  return 0;
}

static void SimulatorDel(CPPClassObject* self)
{
  SimulationEngine* engine = (SimulationEngine*)(self->cpp_obj);
  delete engine;
  self->ob_type->tp_free((PyObject*) self);
}

PyDoc_STRVAR(reset__doc__,
"reset()->\n\n"
"Reset the simulator for new simulations."
);
static PyObject* SimulatorReset(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SimulationEngine* engine = (SimulationEngine*)(cppclass_obj->cpp_obj); 
  engine->ResetEngine();
}

PyDoc_STRVAR(simulate__doc__,
"simulate(start_element, end_element)->\n\n"
"Simulate the beam in the beamline range [start_element, end_element] (inclusive)."
);
static PyObject* SimulatorStart(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SimulationEngine* engine = (SimulationEngine*)(cppclass_obj->cpp_obj); 
  int narg = PyTuple_Size(args);
  if(narg == 0)
  {
    engine->Simulate();
  }
  else if(narg == 2)
  {
    char* start, *end;
    if(!PyArg_ParseTuple(args, "ss:Simulate(A, B)", &start, &end))
      std::cerr << "Simulator::Simulate from A to B needs the names of the starting and ending elements" << std::endl;
    else
      engine->Simulate(std::string(start), std::string(end));
  }
  else 
    std::cerr << "SimulationEngine Start routine needs either 0 or 2 args!" << std::endl;
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(set_space_charge__doc__,
"set_space_charge(option)->\n"
"option: \"on\" or \"off\".\n\n"
"Turn on/off space charge effect."
);
static PyObject* SetSpaceCharge(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  SimulationEngine* engine = (SimulationEngine*)(cppclass_obj->cpp_obj); 
  char* on_off;
  if(!PyArg_ParseTuple(args, "s:set_space_charge(on_off)", &on_off))
  {
    std::cerr << "Simulator::set_space_charge error: argument options are (\"on/off\")" << std::endl;
    return 0;
  }
  std::string on_off_str(on_off);
  std::transform(on_off_str.begin(), on_off_str.end(), on_off_str.begin(), ::tolower);      
  engine->SetSpaceCharge(on_off_str);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef SimulatorMethods[] = {
  {"reset", SimulatorReset, METH_VARARGS, simulate__doc__},
  {"simulate", SimulatorStart, METH_VARARGS, simulate__doc__},
  {"set_space_charge", SetSpaceCharge, METH_VARARGS, set_space_charge__doc__},
  {NULL, NULL}
};

static PyMemberDef SimulatorMembers[] = {
  {NULL}
};

static PyTypeObject Simulator_Type = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "Simulator", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) SimulatorDel, /*tp_dealloc*/
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
    simulator_init__doc__, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    SimulatorMethods, /* tp_methods */
    SimulatorMembers, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) SimulatorInit, /* tp_init */
    0, /* tp_alloc */
    SimulatorNew, /* tp_new */
};

PyMODINIT_FUNC initSimulator(PyObject* module)
{
  if(PyType_Ready(&Simulator_Type) < 0) return;
  Py_INCREF(&Simulator_Type);
  PyModule_AddObject(module, "Simulator", (PyObject*)&Simulator_Type);
}


#ifdef _cplusplus
}
#endif


