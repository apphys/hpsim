#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <Python.h>
#include <arrayobject.h>
#include "wrap_beam.h"
#include "wrap_beamline.h"
#include "wrap_simulator.h"
#include "wrap_dbconnection.h"
#include "wrap_spacecharge.h"
#include "hpsim_module.h"
#include "sql_utility.h"
#include "cppclass_object.h"
#include "beamline.h"
#include "pv_observer_list.h"
#include "init.h"
#include "init_pv_observer_list.h"


#ifdef _cplusplus
extern "C" {
#endif

typedef std::vector<std::string> StringArray;
typedef std::vector<std::pair<std::string, std::string> > StringPairArray;
typedef std::vector<std::pair<std::string, std::pair<std::string, std::string> > >
        StringTripletArray;
namespace
{
  PVObserverList pv_oblist;  
}

PyDoc_STRVAR(set_gpu__doc__,
"set_gpu(device_id) ->\n\n"
"Set which gpu to use for the simulation. Use \"nvidia-smi\" for more info."
);
static PyObject* SetGPU(PyObject* self, PyObject* args)
{
  int gpu_id;
  if(!PyArg_ParseTuple(args, "i", &gpu_id))
  {
    std::cerr << "set_gpu needs a gpu id!" << std::endl;
    return NULL;
  }
  SetGPU(gpu_id);  
  Py_INCREF(Py_None);
  return Py_None;
}

////old
//PyDoc_STRVAR(set_db_epics_old__doc__,
//"set_db_epics(pv_name, value, DBConnection, BeamLine) ->\n\n"
//"Set the epics channel (PV) value."
//);
//static PyObject* SetDbEPICSOld(PyObject* self, PyObject* args)
//{
//  char* pv;
//  char* value;
//  PyObject* py_beamline, *py_dbconnection; 
//  if(!PyArg_ParseTuple(args, "ssOO", &pv, &value, &py_dbconnection, &py_beamline))
//  {
//    std::cerr << "set_db_epics needs a pv, a value, a db connection,and a beamline as args!" << std::endl;
//    return NULL;
//  }
//  std::cout << "set_db_epics(): pv = " << std::string(pv) << ", value = " << value << std::endl;
//  PyObject* py_beamline_type = getHPSimType("BeamLine");
//  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
//  if(PyObject_IsInstance(py_beamline, py_beamline_type) && 
//     PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
//  {
//    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
//    Updater* up = new Updater(std::string(pv));
//    up->SetDBconn(dbconn->db_conn);
//    int dbs_sz = (dbconn->dbs).size();
//    int dbs_indx = 0; 
//    std::string sql;
//    std::string id_str = "";
//    while(dbs_indx < dbs_sz && id_str == "")
//    {
//      sql = "select id from " + (dbconn->dbs)[dbs_indx] + ".epics_channel where lcs_name = '" + std::string(pv) + "'";
//      id_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//      ++dbs_indx;
//    }           
//    if(id_str != "")
//    {
//      --dbs_indx; 
//      up->SetDB((dbconn->dbs)[dbs_indx]);
//      BeamLine* beamline = (BeamLine*)((CPPClassObject*)py_beamline)->cpp_obj;
//      up->SetBeamLine(beamline->GetHostBeamLinePtr());
//      up->SetValue(std::string(value));
//      up->UpdateDB();
//      up->UpdateModel();
//      delete up;
//    }
//    else
//      std::cerr << "Error in set_db_epics : can't find lcs_name " << std::string(pv) << std::endl;
//  }// if PyObject_IsInstance
//
//  Py_INCREF(Py_None);
//  return Py_None;
//}

PyDoc_STRVAR(set_db_epics__doc__,
"set_db_epics(pv_name, value, DBConnection, BeamLine) ->\n\n"
"Set the epics channel (PV) value."
);
static PyObject* SetDbEPICS(PyObject* self, PyObject* args)
{
  char* pv;
  char* value;
  PyObject* py_beamline, *py_dbconnection; 
  if(!PyArg_ParseTuple(args, "ssOO", &pv, &value, &py_dbconnection, &py_beamline))
  {
    std::cerr << "set_db_epics() needs a pv, a value, a db connection,and a beamline as args!" << std::endl;
    return NULL;
  }
  std::cout << "set_db_epics(): pv = " << std::string(pv) << ", value = " << value << std::endl;
  PyObject* py_beamline_type = getHPSimType("BeamLine");
  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
  if(PyObject_IsInstance(py_beamline, py_beamline_type) && 
     PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
  {
    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
    BeamLine* beamline = (BeamLine*)((CPPClassObject*)py_beamline)->cpp_obj;
    if(pv_oblist.GetSize() == 0)
      InitPVObserverList(pv_oblist, *beamline, *dbconn);
    PVObserver* updater = pv_oblist[std::string(pv)];
    if(updater != NULL)
      updater->Update(std::string(value));
  }// if PyObject_IsInstance

  Py_INCREF(Py_None);
  return Py_None;
}

/*!
 * /todo Rewrite this set_db_model() using GetQueryResults(). Does all db 
 * 	 fields appropriately named?
 */
PyDoc_STRVAR(set_db_model__doc__,
"set_db_model(record_name, field_name, value, DBConnection, BeamLine) ->\n\n"
"Set the value of a model parameter in the db."
);
static PyObject* SetDbModel(PyObject* self, PyObject* args)
{
  char* record_name, *field_name, *value;
  PyObject* py_beamline, *py_dbconnection; 
  if(!PyArg_ParseTuple(args, "sssOO", &record_name, &field_name, &value, &py_dbconnection, &py_beamline))
  {
    std::cerr << "set_db_model() needs a record_name, a field_name, a value, a db connection, and a beamline as args!" << std::endl;
    return NULL;
  }
  PyObject* py_beamline_type = getHPSimType("BeamLine");
  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
  if(PyObject_IsInstance(py_beamline, py_beamline_type) && 
     PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
  { 
    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
    BeamLine* bl = (BeamLine*)((CPPClassObject*)py_beamline)->cpp_obj;
    std::string sql;
    for(int dbs_indx = 0; dbs_indx < (dbconn->dbs).size(); ++dbs_indx)
    {
      sql = "select name from " + (dbconn->dbs)[dbs_indx] + ".sqlite_master " +
            "where type = 'table'";  
      StringArray tbl_names = GetDataArrayFromDB(dbconn->db_conn, sql.c_str()); 
      int saindx = 0;
      while(saindx < tbl_names.size())
      {
        // not every table has a name, i.e. epics_channel has lcs_name
        sql = "select * from " + (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx];
        StringArray col_names = GetTableColumnNames(dbconn->db_conn, sql.c_str());
        StringArray::iterator it = std::find(col_names.begin(), col_names.end(), 
                                              std::string(field_name));
        // table has the column/field name
        if(it != col_names.end() && (tbl_names[saindx] == "buncher" || 
          tbl_names[saindx] == "rf_module" || 
          tbl_names[saindx] == "spch_comp" || 
          tbl_names[saindx] == "raperture" || 
          tbl_names[saindx] == "caperture" ))
        {
          sql = "select id from " + (dbconn->dbs)[dbs_indx] + "." + 
            tbl_names[saindx] + " where name = '" + std::string(record_name) + "'";
          std::string value_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
          if(value_str != "") // table has the record name
          {
            sql = "update " + (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx] +
              " set " + std::string(field_name) + " = " + std::string(value) +
              " where name = '" + std::string(record_name) + "'";
            char* errmsg;
            SQLCheck(sqlite3_exec(dbconn->db_conn, sql.c_str(), NULL, NULL, &errmsg), 
                    "set_db_model: sqlite3_exec: " + std::string(record_name) + 
                    ":" + std::string(field_name));
            // change model in the pinned mem
            if(tbl_names[saindx] == "spch_comp" || tbl_names[saindx] == "buncher" ||
               tbl_names[saindx] == "raperture" || tbl_names[saindx] == "caperture")
            {
              sql = "select model_index from " + (dbconn->dbs)[dbs_indx] + "." + 
                tbl_names[saindx] + " where name = '" + std::string(record_name) + "'";
              std::string indx_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
              int model_index = std::atoi(indx_str.c_str());
              if(tbl_names[saindx] == "spch_comp" &&
                  std::string(field_name) == "fraction_model")
              {
                sql = "select fraction_model from " + (dbconn->dbs)[dbs_indx] +
                  ".spch_comp where name = '" + std::string(record_name) + "'";
                std::string frac_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
		SpaceChargeCompensation* elem = dynamic_cast<SpaceChargeCompensation*>((*bl)[model_index]);
		elem->SetFraction(std::atof(frac_str.c_str()));
                //bl[model_index].t = std::atof(frac_str.c_str()); 
              } 
              if(tbl_names[saindx] == "buncher" && std::string(field_name) == "phase_offset_cal")
              {
                sql = "select phase_model from " + (dbconn->dbs)[dbs_indx] + 
                      ".buncher where name = '" + std::string(record_name) + "'";
                std::string phase_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
		Buncher* elem = dynamic_cast<Buncher*>((*bl)[model_index]);
		elem->SetPhase(std::atof(phase_str.c_str()));
                //bl[model_index].phi_c = std::atof(phase_str.c_str());
              }
              if(tbl_names[saindx] == "raperture" && 
                (std::string(field_name) == "aperture_xl_model" || 
                 std::string(field_name) == "aperture_xr_model" || 
                 std::string(field_name) == "aperture_yt_model" || 
                 std::string(field_name) == "aperture_yb_model" || 
                 std::string(field_name) == "in_out_model"))
              {
                
                sql = "select " + std::string(field_name) + " from " + (dbconn->dbs)[dbs_indx] + 
                      ".raperture where name = '" + std::string(record_name) + "'";
                std::string rap_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
		ApertureRectangular* elem = dynamic_cast<ApertureRectangular*>((*bl)[model_index]);	
                if(std::string(field_name) == "aperture_xl_model")
		  elem->SetApertureXLeft(std::atof(rap_str.c_str()));
                  //bl[model_index].aperture1 = std::atof(rap_str.c_str());
                if(std::string(field_name) == "aperture_xr_model")
		  elem->SetApertureXRight(std::atof(rap_str.c_str()));
                  //bl[model_index].tp = std::atof(rap_str.c_str());
                if(std::string(field_name) == "aperture_yt_model")
		  elem->SetApertureYTop(std::atof(rap_str.c_str()));
                  //bl[model_index].aperture2 = std::atof(rap_str.c_str());
                if(std::string(field_name) == "aperture_yb_model")
		  elem->SetApertureYBottom(std::atof(rap_str.c_str()));
                  //bl[model_index].sp = std::atof(rap_str.c_str());
                if(std::string(field_name) == "in_out_model")
		  std::atof(rap_str.c_str()) > 0 ? elem->SetIn() : elem->SetOut();
                  //bl[model_index].t= std::atof(rap_str.c_str());
              }
              if(tbl_names[saindx] == "caperture" && 
                (std::string(field_name) == "aperture_model" || 
                 std::string(field_name) == "in_out_model"))
              {
                
                sql = "select " + std::string(field_name) + " from " + (dbconn->dbs)[dbs_indx] + 
                      ".caperture where name = '" + std::string(record_name) + "'";
                std::string rap_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
		ApertureCircular* elem = dynamic_cast<ApertureCircular*>((*bl)[model_index]);	
                if(std::string(field_name) == "aperture_model")
		  elem->SetAperture(std::atof(rap_str.c_str()));
                  //bl[model_index].aperture1 = std::atof(rap_str.c_str());
                if(std::string(field_name) == "in_out_model")
		  std::atof(rap_str.c_str()) > 0 ? elem->SetIn() : elem->SetOut();
                  //bl[model_index].t= std::atof(rap_str.c_str());
              }
            }
            else if (tbl_names[saindx] == "rf_module")
            {
              if(std::string(field_name) == "phase_offset_cal")
              {
                sql = "select model_index, beam_phase_shift_model from " + 
                  (dbconn->dbs)[dbs_indx] + ".rf_gap where module_id = "
                  "(select id from " + (dbconn->dbs)[dbs_indx] + ".rf_module where name ='" + 
                  std::string(record_name) + "')";
                StringPairArray indx_ph = GetDataPairArrayFromDB(dbconn->db_conn, sql.c_str()); 
                for(int iip = 0; iip < indx_ph.size(); ++iip)
                {
                  int model_index = std::atoi(indx_ph[iip].first.c_str());
		  RFGap* elem = dynamic_cast<RFGap*>((*bl)[model_index]);	
		  elem->SetPhaseShift(std::atof(indx_ph[iip].second.c_str()));
                  //bl[model_index].phi_out = std::atof(indx_ph[iip].second.c_str());
                }
              }
              else if(std::string(field_name) == "amplitude_scale_cal" ||
                std::string(field_name) == "amplitude_tilt_cal" ||
                std::string(field_name) == "voltage_cal")
              {
                sql = "select model_index, amplitude_model, ref_phase_model from " + 
                  (dbconn->dbs)[dbs_indx] + ".rf_gap where module_id = "
                  "(select id from " + (dbconn->dbs)[dbs_indx] + ".rf_module where name ='" + 
                  std::string(record_name) + "')";
                StringTripletArray indx_amp_ph = GetDataTripletArrayFromDB(
                                                  dbconn->db_conn, sql.c_str());
                for(int iiap = 0; iiap < indx_amp_ph.size(); ++iiap)
                {
                  int model_index = std::atoi(indx_amp_ph[iiap].first.c_str());
		  RFGap* elem = dynamic_cast<RFGap*>((*bl)[model_index]);	
		  elem->SetRFAmplitude(std::atof(indx_amp_ph[iiap].second.first.c_str()));
		  elem->SetRefPhase(std::atof(indx_amp_ph[iiap].second.second.c_str()));
                  //bl[model_index].rf_amp = std::atof(indx_amp_ph[iiap].second.first.c_str());
                  //bl[model_index].phi_c = std::atof(indx_amp_ph[iiap].second.second.c_str());
                }
              } // field_name
            } // tbl_names
            Py_INCREF(Py_None);
            return Py_None;
          } // table does not have the record name
          else
            saindx++;
        }
        else // table does not have the column name or it is not the right table
          saindx++;
      } // while
    }// for dbs_indx
    std::cerr << "Error in set_db_model: can't make the change for " 
        << record_name << ":" << field_name << std::endl;  
  }// if PyObject_IsInstance
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_db_epics__doc__,
"get_db_epics(pv_name, DBConnection)-> \n\n"
"Get a EPICS channel value from the db."
);
static PyObject* GetDbEPICS(PyObject* self, PyObject* args)
{
  char* pv_name;
  PyObject* py_dbconnection; 
  if(!PyArg_ParseTuple(args, "sO", &pv_name, &py_dbconnection))
  {
    std::cerr << "get_db_epics needs a pv and a db connection as args!" << std::endl;
    return NULL;
  }
  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
  { 
    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
    std::string sql;
    for(int dbs_indx = 0; dbs_indx < (dbconn->dbs).size(); ++dbs_indx)
    {
      sql = "select value from " + (dbconn->dbs)[dbs_indx] +
             ".epics_channel where lcs_name = '" + std::string(pv_name) + "'";
      StringArray value_trpl = GetDataArrayFromDB(dbconn->db_conn, sql.c_str());
      if(value_trpl.empty()) 
        continue;
      else
      {
        std::string value_str;
        value_str = value_trpl[0];
        return PyString_FromString(value_str.c_str());
      }
    }// for dbs_indx
    std::cerr << "Error in get_db_epics : can't find pv: " << pv_name << ", or this record has a NULL value " << std::endl;  
  }// if PyObject_IsInstance
  Py_INCREF(Py_None);
  return Py_None;
}

PyDoc_STRVAR(get_db_model__doc__,
"get_db_model(record_name, field_name, DBConnection) ->\n\n"
"Get a model parameter value from the db."
);
static PyObject* GetDbModel(PyObject* self, PyObject* args)
{
  char* record_name, *field_name;
  PyObject* py_dbconnection; 
  if(!PyArg_ParseTuple(args, "ssO", &record_name, &field_name, &py_dbconnection))
  {
    std::cerr << "get_db_model needs a record_name, a field_name, and a db "
      "connection as args!" << std::endl;
    return NULL;
  }
  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
  { 
    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
    std::string sql;
    for(int dbs_indx = 0; dbs_indx < (dbconn->dbs).size(); ++dbs_indx)
    {
      sql = "select name from " + (dbconn->dbs)[dbs_indx] + ".sqlite_master " +
            "where type = 'table'";  
      StringArray tbl_names = GetDataArrayFromDB(dbconn->db_conn, sql.c_str()); 
      int saindx = 0;
      while(saindx < tbl_names.size())
      {
        if(tbl_names[saindx] == "epics_channel")
        {
          saindx++;
          continue;
        }
        sql = "select * from " + (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx];
        StringArray col_names = GetTableColumnNames(dbconn->db_conn, sql.c_str());
        StringArray::iterator it = std::find(col_names.begin(), col_names.end(), 
                                              std::string(field_name));
        if(it != col_names.end())
        {
          sql = "select " + std::string(field_name) + " from " + 
            (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx] + 
            " where name = '" + std::string(record_name) + "'";
          std::string value_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
          if(value_str != "")
            return PyString_FromString(value_str.c_str());            
          else
            saindx++;
        }
        else
          saindx++;
      } // while
    }// for dbs_indx
    std::cerr << "Error in get_db_model: can't find record name: " << record_name 
      << " or its field name: " << field_name << std::endl;  
  }// if PyObject_IsInstance
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef HPSimModuleMethods[]={
  {"set_gpu", (PyCFunction)SetGPU, METH_VARARGS, set_gpu__doc__}, 
//  {"set_db_epics", (PyCFunction)SetDbEPICS, METH_VARARGS, set_db_epics__doc__}, 
//  {"set_db_model", (PyCFunction)SetDbModel, METH_VARARGS, set_db_model__doc__}, 
  {"get_db_epics", (PyCFunction)GetDbEPICS, METH_VARARGS, get_db_epics__doc__}, 
  {"get_db_model", (PyCFunction)GetDbModel, METH_VARARGS, get_db_model__doc__}, 
  {NULL}
};

PyMODINIT_FUNC initHPSim()
{
  PyObject* module = Py_InitModule("HPSim", HPSimModuleMethods);
  import_array();
  initBeam(module);
  initDBConnection(module);
  initBeamLine(module);
  initSimulator(module);
  initSpaceCharge(module);
}

PyObject* getHPSimType(char* name)
{
  PyObject* mod = PyImport_ImportModule("HPSim");
  PyObject* pyType = PyObject_GetAttrString(mod, name);
  Py_DECREF(mod);
  Py_DECREF(pyType);
  return pyType;
}
#ifdef _cplusplus
}
#endif
