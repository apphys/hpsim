#include <iostream>
#include <vector>
#include <string>
#include "wrap_beam.h"
#include "beam.h"
#include "cppclass_object.h"

#ifdef _cplusplus
extern "C" {
#endif

static PyObject* BeamNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  CPPClassObject* self = (CPPClassObject*)(type->tp_alloc(type, 0));
  self->cpp_obj = NULL;
  return (PyObject*) self;
}

PyDoc_STRVAR(beam_init__doc__,
"Beam(file = filename)\n"
"Beam(mass = particle_mass, charge = particle_charge, current = beam_current, num = particle_number)\n\n"
"Beam class."
);
static int BeamInit(CPPClassObject* self, PyObject* args, PyObject* kwds)
{
  char* file_name = "";
  double mass, charge, current = 0.0;
  int num = 0;
  static char* kwlist[] = {"mass", "charge", "current", "num", "file", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "|dddis:__init__", kwlist, &mass, 
      &charge, &current, &num, &file_name))
  {
    std::cerr << "Beam constructor arg num does not match!" << std::endl;
    return 0;
  }
  self->cpp_obj = new Beam();
  if(file_name != "")
    ((Beam*)(self->cpp_obj))->InitBeamFromFile(file_name);
  else if(num != 0)
    ((Beam*)(self->cpp_obj))->AllocateBeam(num, mass, charge, current); 
  else
  {
    std::cerr << "Beam constructor need either a file name or num of particles as arg!" << std::endl;
    return 0;
  }
  ((Beam*) self->cpp_obj)->SetWrapper((PyObject*) self);
  return 0;
}

static void BeamDel(CPPClassObject* self)
{
  delete (Beam*)(self->cpp_obj);
  self->ob_type->tp_free((PyObject*) self);
}

PyDoc_STRVAR(set_distribution__doc__,
"set_distribution(x, xp, y, yp, phi, w, loss) -> \n\n"
"set/change beam distribution.");
static PyObject* BeamSetDistribution(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  PyObject *x, *xp, *y, *yp, *phi, *w, *loss = NULL, *lloss = NULL;
  if(!PyArg_ParseTuple(args, "OOOOOO|OO", &x, &xp, &y, &yp, &phi, 
      &w, &loss, &lloss))
  {
    std::cerr << "Beam.set_distribution needs (x, xp, y, yp, phi, w, loss"
      "(optional)), lloss(optional, longitudinal loss) as its arg!" << std::endl;
    return 0;
  }
  if(!PyList_Check(x)|| !PyList_Check(xp) || !PyList_Check(y) || 
     !PyList_Check(yp) || !PyList_Check(phi) || !PyList_Check(w) ||
     (loss != NULL && !PyList_Check(loss)) || 
     (lloss != NULL && !PyList_Check(lloss)))  
  {
    std::cerr << "Beam.set_distribution error: beam coordinates should be "
      "python lists!" << std::endl; 
    return 0;
  }
  int len = PyList_Size(x);
  std::vector<double> x_c(len, 0.0);
  std::vector<double> xp_c(len, 0.0);
  std::vector<double> y_c(len, 0.0);
  std::vector<double> yp_c(len, 0.0);
  std::vector<double> phi_c(len, 0.0);
  std::vector<double> w_c(len, 0.0);
  for(int i = 0; i < PyList_Size(x); ++i) 
  {
    x_c[i] = PyFloat_AsDouble(PyList_GetItem(x, i)); 
    xp_c[i] = PyFloat_AsDouble(PyList_GetItem(xp, i)); 
    y_c[i] = PyFloat_AsDouble(PyList_GetItem(y, i)); 
    yp_c[i] = PyFloat_AsDouble(PyList_GetItem(yp, i)); 
    phi_c[i] = PyFloat_AsDouble(PyList_GetItem(phi, i)); 
    w_c[i] = PyFloat_AsDouble(PyList_GetItem(w, i)); 
  }
  std::vector<uint>* loss_ptr = NULL, *lloss_ptr = NULL;
  if(loss != NULL)
  {
    std::vector<uint> loss_c(len, 0);
    for(int i = 0; i < PyList_Size(x); ++i) 
      loss_c[i] = (uint) PyInt_AsLong(PyList_GetItem(loss, i)); 
    loss_ptr = &loss_c;
  }
  if(lloss != NULL)
  {
    std::vector<uint> lloss_c(len, 0);
    for(int i = 0; i < PyList_Size(x); ++i) 
      lloss_c[i] = (uint) PyInt_AsLong(PyList_GetItem(loss, i)); 
    lloss_ptr = &lloss_c;
  }
  beam->InitBeamFromDistribution(x_c, xp_c, y_c, yp_c, phi_c, w_c, loss_ptr, lloss_ptr);
  Py_INCREF(Py_None);
  return Py_None;
}


PyDoc_STRVAR(set_waterbag__doc__,
"set_waterbag(alph_x, beta_x, emittance_x, alpha_y, beta_y, emittance_y, alpha_z, beta_z, emittance_z, phi, w, frequency, random_seed (optional)) -> \n\n"
"set waterbag beam distribution, input format follows the Parmila standard."
);
static PyObject* BeamSetWaterBag(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  double ax, bx, ex, ay, by, ey, az, bz, ez, phi, w, freq;
  int seed = 0;
  if(!PyArg_ParseTuple(args, "dddddddddddd|i", &ax, &bx, &ex, &ay, &by, &ey, 
      &az, &bz, &ez, &phi, &w, &freq, &seed))
  {
    std::cerr << "Beam.set_waterbag needs (ax, bx, ex, ay, by, ey, az, bz, ez, phi, w, freq, seed (optional)) as its arg!"<< std::endl;
    return NULL;
  }
  beam->InitWaterbagBeam(ax, bx, ex, ay, by, ey, az, bz, ez, phi, w, freq, seed);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(set_dc__doc__,
"set_dc(alph_x, beta_x, emittance_x, alpha_y, beta_y, emittance_y, width_phi, synchronous_phi, synchronous_w, random_seed (optional)) -> \n\n"
"set DC beam distribution, input format follows the Parmila standard."
);
static PyObject* BeamSetDC(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  double ax, bx, ex, ay, by, ey, dphi, sync_phi, sync_w;
  int seed = 0;
  if(!PyArg_ParseTuple(args, "ddddddddd|i", &ax, &bx, &ex, &ay, &by, &ey, 
      &dphi, &sync_phi, &sync_w, &seed))
  {
    std::cerr << "Beam.set_dc needs (ax, bx, ex, ay, by, ey, dphi, sync_phi, sync_w) as its arg!"<< std::endl;
    return NULL;
  }
  beam->InitDCBeam(ax, bx, ex, ay, by, ey, dphi, sync_phi, sync_w, seed);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(save_initial_beam__doc__,
"save_initial_beam() -> \n\n"
"Save the initial beam distribution."
);
static PyObject* BeamSaveInitial(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.save_initial_beam needs no arg!"<< std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->SaveInitialBeam();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(save_intermediate_beam__doc__,
"save_intermediate_beam() -> \n\n"
"Save the intermediate beam distribution."
);
static PyObject* BeamSaveIntermediate(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.save_intermediate_beam needs no arg!"<< std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->SaveIntermediateBeam();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(restore_initial_beam__doc__,
"restore_initial_beam() -> \n\n"
"Restore the beam distribution to its initial setting."
);
static PyObject* BeamRestoreInitial(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.restore_initial_beam needs no arg!"<< std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->RestoreInitialBeam();
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(restore_intermediate_beam__doc__,
"restore_intermediate_beam() -> \n\n"
"Restore the beam distribution to its intermediate setting."
);
static PyObject* BeamRestoreIntermediate(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.restore_intermediate_beam needs no arg!"<< std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->RestoreIntermediateBeam();
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* BeamPrintSimple(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
    return NULL;
  beam->PrintSimple();
  Py_INCREF(Py_None);
  return Py_None;
}


PyDoc_STRVAR(print_to__doc__,
"print_to(output_filename, message(optional)) -> \n\n"
"Output beam distribution into a file. The message (empty by default) will be written on the info (the first) line of the output file."
);
static PyObject* BeamPrintTo(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  char* output_file, *msg = "";
  if(!PyArg_ParseTuple(args, "s|s:CheckOut", &output_file, &msg))
  {
    std::cerr << "Beam.print_to takes a file name as its arg and a msg as "
              "its optional." << std::endl;
    return NULL;
  }
  beam->PrintToFile(std::string(output_file), std::string(msg));
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(set_ref_w__doc__,
"set_ref_w(beam_reference_energy) -> \n\n"
"Set the reference energy for simulation. Mostly used for plotting purpose."
);
static PyObject* BeamSetRefEnergy(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  int narg = PyTuple_Size(args);
  if(narg != 1)
  {
    std::cerr << "Beam.set_ref_w takes one arg!" << std::endl;
    return NULL;
  } 
  double ref_energy;
  if(!PyArg_ParseTuple(args, "d:SetRefEnergy", &ref_energy))
    return NULL;
  beam->SetRefEnergy(ref_energy);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(set_ref_phi__doc__,
"set_ref_phi(beam_reference_phase) -> \n\n"
"Set the reference phase for simulation. Mostly used for plotting purpose."
);
static PyObject* BeamSetRefPhase(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  int narg = PyTuple_Size(args);
  if(narg != 1)
  {
    std::cerr << "Beam.set_ref_phi takes one arg!" << std::endl;
    return NULL;
  } 
  double ref_phase;
  if(!PyArg_ParseTuple(args, "d:SetRefPhase", &ref_phase))
    return NULL;
  beam->SetRefPhase(ref_phase);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(set_frequency__doc__,
"set_frequency(frequency) -> \n\n"
"Set the frequency of the beamline that the beam is in."
);
static PyObject* BeamSetFrequency(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  int narg = PyTuple_Size(args);
  double freq;
  if(narg != 1)
  {
    std::cerr << "Beam.set_frequency takes one arg!" << std::endl;
    return NULL;
  }
  if(!PyArg_ParseTuple(args, "d:SetFrequency", &freq))
    return NULL;
  beam->freq = freq;
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_mass__doc__, 
"get_mass() -> float\n\n"
"Get the mass of the particle."
);
static PyObject* BeamGetMass(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.get_mass needs no args!" << std::endl;
    return NULL;
  }
  return PyFloat_FromDouble(beam->mass);
}

PyDoc_STRVAR(get_charge__doc__, 
"get_charge() -> float\n\n"
"Get the charge of the particle."
);
static PyObject* BeamGetCharge(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.get_charge needs no args!" << std::endl;
    return NULL;
  }
  return PyFloat_FromDouble(beam->charge);
}

PyDoc_STRVAR(get_frequency__doc__, 
"get_frequency() -> float\n\n"
"Get the frequency of the beamline which the beam is currently in."
);
static PyObject* BeamGetFrequency(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam.get_frequency needs no args!" << std::endl;
    return NULL;
  }
  double freq = beam->freq;
  return PyFloat_FromDouble(freq);
}

PyDoc_STRVAR(get_ref_w__doc__, 
"get_ref_w() -> float\n\n"
"Get the reference energy of the beam. "
);
static PyObject* BeamGetRefEnergy(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "beam::get_ref_w needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  return PyFloat_FromDouble(beam->GetRefEnergy());
}

PyDoc_STRVAR(get_ref_phi__doc__, 
"get_ref_phi() -> float\n\n"
"Get the reference phase of the beam. "
);
static PyObject* BeamGetRefPhase(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "beam::get_ref_phi needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  return PyFloat_FromDouble(beam->GetRefPhase());
}

PyDoc_STRVAR(get_size__doc__, 
"get_size() -> int\n\n"
"Get the nunber of particles in the beam."
);
static PyObject* BeamGetSize(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "beam::get_size needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  return PyLong_FromUnsignedLong(beam->num_particle);
}

PyDoc_STRVAR(get_x__doc__,
"get_x() -> list(float)\n\n"
"Get the x coordinates of the beam."
);
static PyObject* BeamGetX(PyObject* self, PyObject* args)
{
  char* option = "good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_x() can have one optional arg, pick from (\"good\", \"lost\", \"all\") ! The default is \"good\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "good" && option_str != "lost" && option_str != "all")
  {
    std::cerr << "Beam::get_x(): invalid option:" << option_str << ", pick from (\"good\", \"lost\", \"all\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  std::vector<uint> loss_arr = beam->GetLoss();
  std::vector<double> w_arr = beam->GetX();
  uint loss_num = beam->GetLossNum();
  uint all_num = beam->num_particle;
  uint arr_sz;
  if(option_str == "all")
     arr_sz = all_num;
  else if(option_str == "good")
    arr_sz = all_num - loss_num;
  else 
    arr_sz = loss_num;
  PyObject *lst = PyList_New(arr_sz);
  if (!lst)
      return NULL;
  uint cnt = 0;
  for (int i = 0; i < all_num; i++) 
  {
    if(option_str == "all")
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, i, num);   
    }
    if(option_str == "good" && loss_arr[i] == 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
    if(option_str == "lost" && loss_arr[i] != 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
  }
  return lst;
}

PyDoc_STRVAR(get_xp__doc__,
"get_xp() -> list(float)\n\n"
"Get the xp coordinates of the beam."
);
static PyObject* BeamGetXp(PyObject* self, PyObject* args)
{
  char* option = "good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_xp() can have one optional arg, pick from (\"good\", \"lost\", \"all\") ! The default is \"good\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "good" && option_str != "lost" && option_str != "all")
  {
    std::cerr << "Beam::get_xp(): invalid option:" << option_str << ", pick from (\"good\", \"lost\", \"all\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  std::vector<uint> loss_arr = beam->GetLoss();
  std::vector<double> w_arr = beam->GetXp();
  uint loss_num = beam->GetLossNum();
  uint all_num = beam->num_particle;
  uint arr_sz;
  if(option_str == "all")
     arr_sz = all_num;
  else if(option_str == "good")
    arr_sz = all_num - loss_num;
  else 
    arr_sz = loss_num;
  PyObject *lst = PyList_New(arr_sz);
  if (!lst)
      return NULL;
  uint cnt = 0;
  for (int i = 0; i < all_num; i++) 
  {
    if(option_str == "all")
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, i, num);   
    }
    if(option_str == "good" && loss_arr[i] == 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
    if(option_str == "lost" && loss_arr[i] != 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
  }
  return lst;
}

PyDoc_STRVAR(get_y__doc__,
"get_y() -> list(float)\n\n"
"Get the y coordinates of the beam."
);
static PyObject* BeamGetY(PyObject* self, PyObject* args)
{
  char* option = "good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_y() can have one optional arg, pick from (\"good\", \"lost\", \"all\") ! The default is \"good\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "good" && option_str != "lost" && option_str != "all")
  {
    std::cerr << "Beam::get_y(): invalid option:" << option_str << ", pick from (\"good\", \"lost\", \"all\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  std::vector<uint> loss_arr = beam->GetLoss();
  std::vector<double> w_arr = beam->GetY();
  uint loss_num = beam->GetLossNum();
  uint all_num = beam->num_particle;
  uint arr_sz;
  if(option_str == "all")
     arr_sz = all_num;
  else if(option_str == "good")
    arr_sz = all_num - loss_num;
  else 
    arr_sz = loss_num;
  PyObject *lst = PyList_New(arr_sz);
  if (!lst)
      return NULL;
  uint cnt = 0;
  for (int i = 0; i < all_num; i++) 
  {
    if(option_str == "all")
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, i, num);   
    }
    if(option_str == "good" && loss_arr[i] == 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
    if(option_str == "lost" && loss_arr[i] != 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
  }
  return lst;
}

PyDoc_STRVAR(get_yp__doc__,
"get_yp() -> list(float)\n\n"
"Get the yp coordinates of the beam."
);
static PyObject* BeamGetYp(PyObject* self, PyObject* args)
{
  char* option = "good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_yp() can have one optional arg, pick from (\"good\", \"lost\", \"all\") ! The default is \"good\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "good" && option_str != "lost" && option_str != "all")
  {
    std::cerr << "Beam::get_yp(): invalid option:" << option_str << ", pick from (\"good\", \"lost\", \"all\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  std::vector<uint> loss_arr = beam->GetLoss();
  std::vector<double> w_arr = beam->GetYp();
  uint loss_num = beam->GetLossNum();
  uint all_num = beam->num_particle;
  uint arr_sz;
  if(option_str == "all")
     arr_sz = all_num;
  else if(option_str == "good")
    arr_sz = all_num - loss_num;
  else 
    arr_sz = loss_num;
  PyObject *lst = PyList_New(arr_sz);
  if (!lst)
      return NULL;
  uint cnt = 0;
  for (int i = 0; i < all_num; i++) 
  {
    if(option_str == "all")
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, i, num);   
    }
    if(option_str == "good" && loss_arr[i] == 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
    if(option_str == "lost" && loss_arr[i] != 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
  }
  return lst;
}

PyDoc_STRVAR(get_phi__doc__,
"get_phi() -> list(float)\n\n"
"Get the phi coordinates of the beam."
);
static PyObject* BeamGetPhi(PyObject* self, PyObject* args)
{
  char* option = "good";
  char* option2 = "absolute";
  if(!PyArg_ParseTuple(args, "|ss", &option, &option2))
  {
    std::cerr << "Beam::get_phi() can have two optional args. "
      "The first option can be (\"good\", \"lost\", \"all\"), "
      "the default is \"good\". The second option can be "
      "(\"absolute\", \"relative\"), the default is \"absolute\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  std::string option_str2 = std::string(option2);
  if(option_str != "good" && option_str != "lost" && option_str != "all")
  {
    std::cerr << "Beam::get_phi(): invalid 1st option:" << option_str << ", pick from (\"good\", \"lost\", \"all\") !" << std::endl;
    return NULL;
  }
  if(option_str2 != "absolute" && option_str2 != "relative")
  {
    std::cerr << "Beam::get_phi(): invalid 2nd option:" << option_str2 << ", pick from (\"absolute\", \"relative\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  std::vector<uint> loss_arr = beam->GetLoss();
  std::vector<double> p_arr;
  if(option_str2 == "absolute")
    p_arr = beam->GetPhi();
  else
  {
    beam->UpdateRelativePhi();
    p_arr = beam->GetRelativePhi();
  }
  uint loss_num = beam->GetLossNum();
  uint all_num = beam->num_particle;
  uint arr_sz;
  if(option_str == "all")
     arr_sz = all_num;
  else if(option_str == "good")
    arr_sz = all_num - loss_num;
  else 
    arr_sz = loss_num;
  PyObject *lst = PyList_New(arr_sz);
  if (!lst)
      return NULL;
  uint cnt = 0;
  for (int i = 0; i < all_num; i++) 
  {
    if(option_str == "all")
    {
      PyObject* num = PyFloat_FromDouble(p_arr[i]);
      PyList_SET_ITEM(lst, i, num);   
    }
    if(option_str == "good" && loss_arr[i] == 0)
    {
      PyObject* num = PyFloat_FromDouble(p_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
    if(option_str == "lost" && loss_arr[i] != 0)
    {
      PyObject* num = PyFloat_FromDouble(p_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
  }
  return lst;
}

PyDoc_STRVAR(get_w__doc__,
"get_w() -> list(float)\n\n"
"Get the w coordinates of the beam."
);
static PyObject* BeamGetW(PyObject* self, PyObject* args)
{
  char* option = "good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_w() can have one optional arg, pick from (\"good\", \"lost\", \"all\") ! The default is \"good\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "good" && option_str != "lost" && option_str != "all")
  {
    std::cerr << "Beam::get_w(): invalid option:" << option_str << ", pick from (\"good\", \"lost\", \"all\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  std::vector<uint> loss_arr = beam->GetLoss();
  std::vector<double> w_arr = beam->GetW();
  uint loss_num = beam->GetLossNum();
  uint all_num = beam->num_particle;
  uint arr_sz;
  if(option_str == "all")
     arr_sz = all_num;
  else if(option_str == "good")
    arr_sz = all_num - loss_num;
  else 
    arr_sz = loss_num;
  PyObject *lst = PyList_New(arr_sz);
  if (!lst)
      return NULL;
  uint cnt = 0;
  for (int i = 0; i < all_num; i++) 
  {
    if(option_str == "all")
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, i, num);   
    }
    if(option_str == "good" && loss_arr[i] == 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
    if(option_str == "lost" && loss_arr[i] != 0)
    {
      PyObject* num = PyFloat_FromDouble(w_arr[i]);
      PyList_SET_ITEM(lst, cnt, num);   
      ++cnt;
    }
  }
  return lst;
}

PyDoc_STRVAR(get_losses__doc__,
"get_losses() -> list(float)\n\n"
"Get the loss coordinates of the beam."
);
static PyObject* BeamGetLoss(PyObject* self, PyObject* args)
{
  char* option = "t";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_losses() takes one optional arg, pick from (\"t\" (transverse), \"l\"(longitudinal)) ! The default is \"t\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "t" && option_str != "l")
  {
    std::cerr << "Beam::get_losses(): invalid option:" << option_str << ", pick from (\"t\", \"l\") !" << std::endl;
    return NULL;
  }

  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  std::vector<uint> array;
  if (option_str == "t")
  {
    beam->UpdateLoss();
    array = beam->GetLoss();
  }
  else 
  {
    beam->UpdateLongitudinalLoss();
    array = beam->GetLongitudinalLoss();
  }
  PyObject *lst = PyList_New(beam->num_particle);
  if (!lst)
      return NULL;
  for (int i = 0; i < beam->num_particle; i++) 
  {
    PyObject *num = PyLong_FromUnsignedLong(array[i]);
    if (!num) 
    {
      Py_DECREF(lst);
      return NULL;
    }
    PyList_SET_ITEM(lst, i, num);   // reference to num stolen
  }
  return lst;
}

PyDoc_STRVAR(get_avg_x__doc__,
"get_avg_x(option(optional)) -> float\n"
"option: \"transverse_good\" (default) or \"longitudinal_good\"\n\n"
"Get the center of the beam in x. \n"
"With \"transverse_good\" option, returns average using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns average using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetAvgX(PyObject* self, PyObject* args)
{
  char* option = "transverse_good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_avg_x() takes one optional arg, pick from (\"transverse_good\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  if (option_str == "transverse_good")
  {
    beam->UpdateAvgX();
    return PyFloat_FromDouble(beam->GetAvgX());
  }
  else
  {
    beam->UpdateAvgX(true);
    return PyFloat_FromDouble(beam->GetAvgX(true));
  }
}

PyDoc_STRVAR(get_avg_y__doc__,
"get_avg_y(option(optional)) -> float\n"
"option: \"transverse_good\" (default) or \"longitudinal_good\"\n\n"
"Get the center of the beam in y. \n"
"With \"transverse_good\" option, returns average using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns average using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetAvgY(PyObject* self, PyObject* args)
{
  char* option = "transverse_good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_avg_y() takes one optional arg, pick from (\"transverse_good\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);

  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  if (option_str == "transverse_good")
  {
    beam->UpdateAvgY();
    return PyFloat_FromDouble(beam->GetAvgY());
  }
  else
  {
    beam->UpdateAvgY(true);
    return PyFloat_FromDouble(beam->GetAvgY(true));
  }
}
PyDoc_STRVAR(get_avg_phi__doc__,
"get_avg_phi(option(optional)) -> float\n"
"option: \"absolute\" (default) or \"relative\" or \"longitudinal_good\"\n\n"
"Get the center of the beam in phi. \n"
"With \"absolute\" option, returns average absolute phase of the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"relative\" option, returns average relative phase (relative to reference particle's phase) of the particles that are not lost. \n"
"With \"longitudinal_good\" option, returns average absolute phase of the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetAvgPhi(PyObject* self, PyObject* args)
{
  char* option = "absolute";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_avg_phi can take one optional arg, it can be "
      "(\"absolute\", \"relative\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  double phi_avg;
  if(option_str == "absolute")
  {
    beam->UpdateAvgPhi();
    phi_avg = beam->GetAvgPhi();
  }
  else if (option_str == "relative")
  {
    beam->UpdateRelativePhi();
    beam->UpdateAvgRelativePhi();
    phi_avg = beam->GetAvgRelativePhi();
  }
  else  // wlloss
  {
    beam->UpdateGoodParticleCount();
    beam->UpdateAvgPhi(true);
    phi_avg = beam->GetAvgPhi(true);
  }
  return PyFloat_FromDouble(phi_avg);
}

PyDoc_STRVAR(get_avg_w__doc__,
"get_avg_w(option(optional)) -> float\n"
"option: \"transverse_good\" (default) or \"longitudinal_good\"\n\n"
"Get the center of the beam in w. \n"
"With \"transverse_good\" option, returns average using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns average using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetAvgW(PyObject* self, PyObject* args)
{
  char* option = "transverse_good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_avg_w() takes one optional arg, pick from (\"transverse_good\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);

  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  if (option_str == "transverse_good")
  {
    beam->UpdateAvgW();
    return PyFloat_FromDouble(beam->GetAvgW());
  }
  else
  {
    beam->UpdateAvgW(true);
    return PyFloat_FromDouble(beam->GetAvgW(true));
  }
}

PyDoc_STRVAR(get_sig_x__doc__,
"get_sig_x(option(optional)) -> float\n"
"option: \"transverse_good\" (default) or \"longitudinal_good\"\n\n"
"Get the center of the beam in x. \n"
"With \"transverse_good\" option, returns std using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns std using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetSigX(PyObject* self, PyObject* args)
{
  char* option = "transverse_good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_sig_x() takes one optional arg, pick from (\"transverse_good\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  if (option_str == "transverse_good")
  {
    beam->UpdateSigX();
    return PyFloat_FromDouble(beam->GetSigX());
  }
  else
  {
    beam->UpdateSigX(true);
    return PyFloat_FromDouble(beam->GetSigX(true));
  }
}

PyDoc_STRVAR(get_sig_y__doc__,
"get_sig_y(option(optional)) -> float\n"
"option: \"transverse_good\" (default) or \"longitudinal_good\"\n\n"
"Get the center of the beam in y. \n"
"With \"transverse_good\" option, returns std using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns std using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetSigY(PyObject* self, PyObject* args)
{
  char* option = "transverse_good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_sig_y() takes one optional arg, pick from (\"transverse_good\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  if (option_str == "transverse_good")
  {
    beam->UpdateSigY();
    return PyFloat_FromDouble(beam->GetSigY());
  }
  else
  {
    beam->UpdateSigY(true);
    return PyFloat_FromDouble(beam->GetSigY(true));
  }
}

PyDoc_STRVAR(get_sig_phi__doc__,
"get_sig_phi(option(optional)) -> float\n"
"option: \"absolute\" (default) or \"relative\" or \"longitudinal_good\"\n\n"
"Get the center of the beam in phi. \n"
"With \"absolute\" option, returns the phase std using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"relative\" option, returns the std of the relative phase (relative to the reference phase) using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns std of the absolute phase using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetSigPhi(PyObject* self, PyObject* args)
{
  char* option = "absolute";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_sig_phi can take one optional arg, it can be "
      "(\"absolute\", \"relative\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  double phi_sig;
  if(option_str == "absolute")
  {
    beam->UpdateSigPhi();
    phi_sig = beam->GetSigPhi();
  }
  else if (option_str == "relative")
  {
    beam->UpdateRelativePhi();
    beam->UpdateSigRelativePhi();
    phi_sig = beam->GetSigRelativePhi();
  }
  else // longitudinal_good
  {
    beam->UpdateGoodParticleCount();
    beam->UpdateAvgPhi(true);
    beam->UpdateRelativePhi(true);
    beam->UpdateSigRelativePhi(true);
    phi_sig = beam->GetSigRelativePhi(true);
  }
  return PyFloat_FromDouble(phi_sig);
}

PyDoc_STRVAR(get_sig_w__doc__,
"get_sig_w(option(optional)) -> float\n"
"option: \"transverse_good\" (default) or \"longitudinal_good\"\n\n"
"Get the center of the beam in w. \n"
"With \"transverse_good\" option, returns std using the particles that are not lost transversely (might be lost longitudinally). \n"
"With \"longitudinal_good\" option, returns std using the particles that are in bucket (not lost neither transversely nor longitudinally). "
);
static PyObject* BeamGetSigW(PyObject* self, PyObject* args)
{
  char* option = "transverse_good";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_sig_w() takes one optional arg, pick from (\"transverse_good\", \"longitudinal_good\")" << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  if (option_str == "transverse_good")
  {
    beam->UpdateSigW();
    return PyFloat_FromDouble(beam->GetSigW());
  }
  else
  {
    beam->UpdateSigW(true);
    return PyFloat_FromDouble(beam->GetSigW(true));
  }
}

PyDoc_STRVAR(get_emittance_x__doc__,
"get_emittance_x__doc__() -> \n\n"
"Get the emittance in x."
);
static PyObject* BeamGetEmitX(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam::get_emittance_x needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  beam->UpdateEmittance();
  double emit_x = beam->GetEmittanceX();
  return PyFloat_FromDouble(emit_x);
}

PyDoc_STRVAR(get_emittance_y__doc__,
"get_emittance_y__doc__() -> \n\n"
"Get the emittance in y."
);
static PyObject* BeamGetEmitY(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam::get_emittance_y needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  beam->UpdateEmittance();
  double emit_y = beam->GetEmittanceY();
  return PyFloat_FromDouble(emit_y);
}

PyDoc_STRVAR(get_emittance_z__doc__,
"get_emittance_z__doc__() -> \n\n"
"Get the longitudinal emittance ."
);
static PyObject* BeamGetEmitZ(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam::get_emittance_z needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  beam->UpdateEmittance();
  double emit_z = beam->GetEmittanceZ();
  return PyFloat_FromDouble(emit_z);
}

PyDoc_STRVAR(get_loss_num__doc__, 
"get_loss_num() -> \n\n"
"Get the number of particles that are lost transversely."
);
static PyObject* BeamGetLossNum(PyObject* self, PyObject* args)
{
  char* option = "t";
  if(!PyArg_ParseTuple(args, "|s", &option))
  {
    std::cerr << "Beam::get_loss_num() take one optional arg, pick from (\"t\" (transverse), \"l\"(longitudinal), 'all') ! The default is \"t\"." << std::endl;
    return NULL;
  }
  std::string option_str = std::string(option);
  if(option_str != "t" && option_str != "l" && option_str != "all")
  {
    std::cerr << "Beam::get_loss_num(): invalid option:" << option_str << ", pick from (\"t\", \"l\", \"all\") !" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  uint lnum; 
  if(option_str == "t")
  {
    beam->UpdateLoss();
    lnum = beam->GetLossNum();
  }
  else if(option_str == "l")
  {
    beam->UpdateLongitudinalLoss();
    lnum = beam->GetLongitudinalLossNum();
  }
  else // "all"
  {
    beam->UpdateLoss();
    beam->UpdateLongitudinalLoss();
    beam->UpdateGoodParticleCount();
    lnum = beam->num_particle - beam->GetGoodParticleNum();
  }
  return PyLong_FromUnsignedLong(lnum);
}

PyDoc_STRVAR(get_current__doc__, 
"get_current() ->\n\n"
"Get the beam current."
);
static PyObject* BeamGetCurrent(PyObject* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, ""))
  {
    std::cerr << "Beam::get_current needs no args!" << std::endl;
    return NULL;
  }
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  beam->UpdateLoss();
  double real_current = beam->current*(1.0-((double)beam->GetLossNum())/((double)beam->num_particle));
  return PyFloat_FromDouble(real_current);
}

//static PyObject* BeamGetAvgR2(PyObject* self, PyObject* args)
//{
//  if(!PyArg_ParseTuple(args, ""))
//    return NULL;
//  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
//  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
//  beam->UpdateLoss();
//  beam->UpdateSigR();
//  double sr2 = beam->GetSigR();
//  return PyFloat_FromDouble(sr2);
//}

PyDoc_STRVAR(translate__doc__, 
"translate(axis, val) ->\n"
"axis : \"x\" or \"xp\" or \"y\" or \"yp\" or \"phi\" or \"w\"\n\n"
"Shift the beam along an axis."
);
static PyObject* BeamTranslate(PyObject* self, PyObject* args)
{
  int narg = PyTuple_Size(args);
  if(narg != 2)
  {
    std::cerr << "Beam.translate takes two args! (axis, val)" << std::endl;
    return NULL;
  } 
  double val;
  char* axis = "";
  if(!PyArg_ParseTuple(args, "sd", &axis, &val))
    return NULL;
  
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  if(std::string(axis) == "x" || std::string(axis) == "X")
    beam->ShiftX(val);
  if(std::string(axis) == "xp" || std::string(axis) == "XP")
    beam->ShiftXp(val);
  if(std::string(axis) == "y" || std::string(axis) == "Y")
    beam->ShiftY(val);
  if(std::string(axis) == "yp" || std::string(axis) == "YP")
    beam->ShiftYp(val);
  if(std::string(axis) == "phi" || std::string(axis) == "PHI")
    beam->ShiftPhi(val);
  if(std::string(axis) == "w" || std::string(axis) == "W")
    beam->ShiftW(val);
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(apply_cut__doc__,
"apply_cut(axis, min, max) -> \n"
"axis : \"x\" or \"y\" or \"p\" or \"w\"\n\n"
"Cut the beam along an axis. Only the particles that are within the range [min, max] can pass."
);
static PyObject* BeamApplyCut(PyObject* self, PyObject* args)
{
  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
  int narg = PyTuple_Size(args);
  if(narg != 3)
  {
    std::cerr << "Beam.apply_cut takes three args! (coordinate, min, max)" << std::endl;
    return NULL;
  } 
  char coord;
  double min, max;
  if(!PyArg_ParseTuple(args, "cdd:ApplyCut", &coord, &min, &max))
    return NULL;
  beam->ApplyCut(coord, min, max);
  Py_INCREF(Py_None);
  return Py_None;
}

//static PyObject* BeamInitPhiAvgWithLloss(PyObject* self, PyObject* args)
//{
//  if(!PyArg_ParseTuple(args, ""))
//  {
//    std::cerr << "Beam::init_phi_avg_with_lloss needs no args!" << std::endl;
//    return NULL;
//  }
//  CPPClassObject* cppclass_obj = (CPPClassObject*)self;
//  Beam* beam = (Beam*)(cppclass_obj->cpp_obj); 
//  beam->InitPhiAvgGood();
//  return Py_None;
//}

static PyMethodDef BeamMethods[] = {
  {"set_distribution", BeamSetDistribution, METH_VARARGS, set_distribution__doc__},
  {"set_waterbag", BeamSetWaterBag, METH_VARARGS, set_waterbag__doc__},
  {"set_dc", BeamSetDC, METH_VARARGS, set_dc__doc__},
  {"save_initial_beam", BeamSaveInitial, METH_VARARGS, save_initial_beam__doc__},
  {"save_intermediate_beam", BeamSaveIntermediate, METH_VARARGS, save_intermediate_beam__doc__},
  {"restore_initial_beam", BeamRestoreInitial, METH_VARARGS, restore_initial_beam__doc__},
  {"restore_intermediate_beam", BeamRestoreIntermediate, METH_VARARGS, restore_intermediate_beam__doc__},
//  {"print_simple", BeamPrintSimple, METH_VARARGS, "PrintSimple routine in Beam"},
  {"print_to", BeamPrintTo, METH_VARARGS, print_to__doc__},
  {"set_ref_w", BeamSetRefEnergy, METH_VARARGS, set_ref_w__doc__},
  {"set_ref_phi", BeamSetRefPhase, METH_VARARGS, set_ref_phi__doc__},
  {"set_frequency", BeamSetFrequency, METH_VARARGS, set_frequency__doc__},
  {"get_mass", BeamGetMass, METH_VARARGS, get_mass__doc__},
  {"get_charge", BeamGetCharge, METH_VARARGS, get_charge__doc__},
  {"get_frequency", BeamGetFrequency, METH_VARARGS, get_frequency__doc__},
  {"get_ref_w", BeamGetRefEnergy, METH_VARARGS, get_ref_w__doc__},
  {"get_ref_phi", BeamGetRefPhase, METH_VARARGS, get_ref_phi__doc__},
  {"get_size", BeamGetSize, METH_VARARGS, get_size__doc__},
  {"get_x", BeamGetX, METH_VARARGS, get_x__doc__},
  {"get_xp", BeamGetXp, METH_VARARGS, get_xp__doc__},
  {"get_y", BeamGetY, METH_VARARGS, get_y__doc__},
  {"get_yp", BeamGetYp, METH_VARARGS, get_yp__doc__},
  {"get_phi", BeamGetPhi, METH_VARARGS, get_phi__doc__},
  {"get_w", BeamGetW, METH_VARARGS, get_w__doc__},
  {"get_losses", BeamGetLoss, METH_VARARGS, get_losses__doc__},
  {"get_loss_num", BeamGetLossNum, METH_VARARGS, get_loss_num__doc__},
  {"get_avg_x", BeamGetAvgX, METH_VARARGS, get_avg_x__doc__},
  {"get_avg_y", BeamGetAvgY, METH_VARARGS, get_avg_y__doc__},
  {"get_avg_phi", BeamGetAvgPhi, METH_VARARGS, get_avg_phi__doc__},
  {"get_avg_w", BeamGetAvgW, METH_VARARGS, get_avg_w__doc__},
  {"get_sig_x", BeamGetSigX, METH_VARARGS, get_sig_x__doc__},
  {"get_sig_y", BeamGetSigY, METH_VARARGS, get_sig_y__doc__},
  {"get_sig_phi", BeamGetSigPhi, METH_VARARGS, get_sig_phi__doc__},
  {"get_sig_w", BeamGetSigW, METH_VARARGS, get_sig_w__doc__},
  {"get_emittance_x", BeamGetEmitX, METH_VARARGS, get_emittance_x__doc__},
  {"get_emittance_y", BeamGetEmitY, METH_VARARGS, get_emittance_y__doc__},
  {"get_emittance_z", BeamGetEmitZ, METH_VARARGS, get_emittance_z__doc__},
  {"get_current", BeamGetCurrent, METH_VARARGS, get_current__doc__},
  {"apply_cut", BeamApplyCut, METH_VARARGS, apply_cut__doc__},
  {"translate", BeamTranslate, METH_VARARGS, translate__doc__},
//  {"init_phi_avg_with_lloss", BeamInitPhiAvgWithLloss, METH_VARARGS, "ShiftW in Beam"},
  {NULL}
};

static PyMemberDef BeamMembers[] = {
  {NULL}
};

static PyTypeObject Beam_Type = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "Beam", /*tp_name*/
    sizeof(CPPClassObject), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor) BeamDel, /*tp_dealloc*/
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
    beam_init__doc__, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    BeamMethods, /* tp_methods */
    BeamMembers, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc) BeamInit, /* tp_init */
    0, /* tp_alloc */
    BeamNew, /* tp_new */
};

PyMODINIT_FUNC initBeam(PyObject* module)
{
  if(PyType_Ready(&Beam_Type) < 0) return;
  Py_INCREF(&Beam_Type);
  PyModule_AddObject(module, "Beam", (PyObject*)&Beam_Type);
}


#ifdef _cplusplus
}
#endif


