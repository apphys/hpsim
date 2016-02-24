#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
//#include <arrayobject.h>
#include "wrap_beam.h"
//#include "wrap_beamline.h"
//#include "wrap_simulator.h"
//#include "wrap_dbconnection.h"
//#include "wrap_spacecharge.h"
#include "hpsim_module.h"
//#include "updater.h"
#include "sql_utility.h"
#include "cppclass_object.h"
//#include "beamline.h"
//#include "init.h"


#ifdef _cplusplus
extern "C" {
#endif

typedef std::vector<std::string> StringArray;
typedef std::vector<std::pair<std::string, std::string> > StringPairArray;
typedef std::vector<std::pair<std::string, std::pair<std::string, std::string> > >
        StringTripletArray;

//PyDoc_STRVAR(set_gpu__doc__,
//"set_gpu(device_id) ->\n\n"
//"Set which gpu to use for the simulation. Use \"nvidia-smi\" for more info."
//);
//static PyObject* SetGPU(PyObject* self, PyObject* args)
//{
//  int gpu_id;
//  if(!PyArg_ParseTuple(args, "i", &gpu_id))
//  {
//    std::cerr << "set_gpu needs a gpu id!" << std::endl;
//    return NULL;
//  }
//  SetGPU(gpu_id);  
//  Py_INCREF(Py_None);
//  return Py_None;
//}
//
//PyDoc_STRVAR(set_db_epics__doc__,
//"set_db_epics(pv_name, value, DBConnection, BeamLine) ->\n\n"
//"Set the epics channel (PV) value."
//);
//static PyObject* SetDbEPICS(PyObject* self, PyObject* args)
//{
//  char* pv;
//  char* value;
//  PyObject* py_beamline, *py_dbconnection; 
//  if(!PyArg_ParseTuple(args, "ssOO", &pv, &value, &py_dbconnection, &py_beamline))
//  {
//    std::cerr << "set_db_epics needs a pv, a value, a db connection,and a beamline as args!" << std::endl;
//    return NULL;
//  }
//  std::cout << "set_db_epics_channel: pv = " << std::string(pv) << ", value = " << value << std::endl;
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
//
//PyDoc_STRVAR(set_db_model__doc__,
//"set_db_model(record_name, field_name, value, DBConnection, BeamLine) ->\n\n"
//"Set the value of a model parameter in the db."
//);
//static PyObject* SetDbModel(PyObject* self, PyObject* args)
//{
//  char* record_name, *field_name, *value;
//  PyObject* py_beamline, *py_dbconnection; 
//  if(!PyArg_ParseTuple(args, "sssOO", &record_name, &field_name, &value, &py_dbconnection, &py_beamline))
//  {
//    std::cerr << "set_db_model needs a record_name, a field_name, a value, a db connection, and a beamline as args!" << std::endl;
//    return NULL;
//  }
//  PyObject* py_beamline_type = getHPSimType("BeamLine");
//  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
//  if(PyObject_IsInstance(py_beamline, py_beamline_type) && 
//     PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
//  { 
//    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
//    BeamLine* beamline = (BeamLine*)((CPPClassObject*)py_beamline)->cpp_obj;
//    BeamLineElement* bl = beamline->GetHostBeamLinePtr();
//    std::string sql;
//    for(int dbs_indx = 0; dbs_indx < (dbconn->dbs).size(); ++dbs_indx)
//    {
//      sql = "select name from " + (dbconn->dbs)[dbs_indx] + ".sqlite_master " +
//            "where type = 'table'";  
//      StringArray tbl_names = GetDataArrayFromDB(dbconn->db_conn, sql.c_str()); 
//      int saindx = 0;
//      while(saindx < tbl_names.size())
//      {
//        // not every table has a name, i.e. epics_channel has lcs_name
//        sql = "select * from " + (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx];
//        StringArray col_names = GetTableColumnNames(dbconn->db_conn, sql.c_str());
//        StringArray::iterator it = std::find(col_names.begin(), col_names.end(), 
//                                              std::string(field_name));
//        // table has the column/field name
//        if(it != col_names.end() && (tbl_names[saindx] == "buncher" || 
//          tbl_names[saindx] == "rf_module" || 
//          tbl_names[saindx] == "spch_comp" || 
//          tbl_names[saindx] == "raperture" || 
//          tbl_names[saindx] == "caperture" ))
//        {
//          sql = "select id from " + (dbconn->dbs)[dbs_indx] + "." + 
//            tbl_names[saindx] + " where name = '" + std::string(record_name) + "'";
//          std::string value_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//          if(value_str != "") // table has the record name
//          {
//            sql = "update " + (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx] +
//              " set " + std::string(field_name) + " = " + std::string(value) +
//              " where name = '" + std::string(record_name) + "'";
//            char* errmsg;
//            SQLCheck(sqlite3_exec(dbconn->db_conn, sql.c_str(), NULL, NULL, &errmsg), 
//                    "set_db_model: sqlite3_exec: " + std::string(record_name) + 
//                    ":" + std::string(field_name));
//            // change model in the pinned mem
//            if(tbl_names[saindx] == "spch_comp" || tbl_names[saindx] == "buncher" ||
//               tbl_names[saindx] == "raperture" || tbl_names[saindx] == "caperture")
//            {
//              sql = "select model_index from " + (dbconn->dbs)[dbs_indx] + "." + 
//                tbl_names[saindx] + " where name = '" + std::string(record_name) + "'";
//              std::string indx_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//              int model_index = std::atoi(indx_str.c_str());
//              if(tbl_names[saindx] == "spch_comp" &&
//                  std::string(field_name) == "fraction_model")
//              {
//                sql = "select fraction_model from " + (dbconn->dbs)[dbs_indx] +
//                  ".spch_comp where name = '" + std::string(record_name) + "'";
//                std::string frac_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//                bl[model_index].t = std::atof(frac_str.c_str()); 
//              } 
//              if(tbl_names[saindx] == "buncher" && std::string(field_name) == "phase_offset_cal")
//              {
//                sql = "select phase_model from " + (dbconn->dbs)[dbs_indx] + 
//                      ".buncher where name = '" + std::string(record_name) + "'";
//                std::string phase_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//                bl[model_index].phi_c = std::atof(phase_str.c_str());
//              }
//              if(tbl_names[saindx] == "raperture" && 
//                (std::string(field_name) == "aperture_xl_model" || 
//                 std::string(field_name) == "aperture_xr_model" || 
//                 std::string(field_name) == "aperture_yt_model" || 
//                 std::string(field_name) == "aperture_yb_model" || 
//                 std::string(field_name) == "in_out_model"))
//              {
//                
//                sql = "select " + std::string(field_name) + " from " + (dbconn->dbs)[dbs_indx] + 
//                      ".raperture where name = '" + std::string(record_name) + "'";
//                std::string rap_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//                if(std::string(field_name) == "aperture_xl_model")
//                  bl[model_index].aperture1 = std::atof(rap_str.c_str());
//                if(std::string(field_name) == "aperture_xr_model")
//                  bl[model_index].tp = std::atof(rap_str.c_str());
//                if(std::string(field_name) == "aperture_yt_model")
//                  bl[model_index].aperture2 = std::atof(rap_str.c_str());
//                if(std::string(field_name) == "aperture_yb_model")
//                  bl[model_index].sp = std::atof(rap_str.c_str());
//                if(std::string(field_name) == "in_out_model")
//                  bl[model_index].t= std::atof(rap_str.c_str());
//              }
//              if(tbl_names[saindx] == "caperture" && 
//                (std::string(field_name) == "aperture_model" || 
//                 std::string(field_name) == "in_out_model"))
//              {
//                
//                sql = "select " + std::string(field_name) + " from " + (dbconn->dbs)[dbs_indx] + 
//                      ".caperture where name = '" + std::string(record_name) + "'";
//                std::string rap_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//                if(std::string(field_name) == "aperture_model")
//                  bl[model_index].aperture1 = std::atof(rap_str.c_str());
//                if(std::string(field_name) == "in_out_model")
//                  bl[model_index].t= std::atof(rap_str.c_str());
//              }
//            }
//            else if (tbl_names[saindx] == "rf_module")
//            {
//              if(std::string(field_name) == "phase_offset_cal")
//              {
//                sql = "select model_index, beam_phase_shift_model from " + 
//                  (dbconn->dbs)[dbs_indx] + ".rf_gap where module_id = "
//                  "(select id from " + (dbconn->dbs)[dbs_indx] + ".rf_module where name ='" + 
//                  std::string(record_name) + "')";
//                StringPairArray indx_ph = GetDataPairArrayFromDB(dbconn->db_conn, sql.c_str()); 
//                for(int iip = 0; iip < indx_ph.size(); ++iip)
//                {
//                  int model_index = std::atoi(indx_ph[iip].first.c_str());
//                  bl[model_index].phi_out = std::atof(indx_ph[iip].second.c_str());
//                }
//              }
//              else if(std::string(field_name) == "amplitude_scale_cal" ||
//                std::string(field_name) == "amplitude_tilt_cal" ||
//                std::string(field_name) == "voltage_cal")
//              {
//                sql = "select model_index, amplitude_model, ref_phase_model from " + 
//                  (dbconn->dbs)[dbs_indx] + ".rf_gap where module_id = "
//                  "(select id from " + (dbconn->dbs)[dbs_indx] + ".rf_module where name ='" + 
//                  std::string(record_name) + "')";
//                StringTripletArray indx_amp_ph = GetDataTripletArrayFromDB(
//                                                  dbconn->db_conn, sql.c_str());
//                for(int iiap = 0; iiap < indx_amp_ph.size(); ++iiap)
//                {
//                  int model_index = std::atoi(indx_amp_ph[iiap].first.c_str());
//                  bl[model_index].rf_amp = std::atof(indx_amp_ph[iiap].second.first.c_str());
//                  bl[model_index].phi_c = std::atof(indx_amp_ph[iiap].second.second.c_str());
//                }
//              } // field_name
//            } // tbl_names
//            Py_INCREF(Py_None);
//            return Py_None;
//          } // table does not have the record name
//          else
//            saindx++;
//        }
//        else // table does not have the column name or it is not the right table
//          saindx++;
//      } // while
//    }// for dbs_indx
//    std::cerr << "Error in set_db_model: can't make the change for " 
//        << record_name << ":" << field_name << std::endl;  
//  }// if PyObject_IsInstance
//  Py_INCREF(Py_None);
//  return Py_None;
//}
//
//PyDoc_STRVAR(get_db_epics__doc__,
//"get_db_epics(pv_name, DBConnection)-> \n\n"
//"Get a EPICS channel value from the db."
//);
//static PyObject* GetDbEPICS(PyObject* self, PyObject* args)
//{
//  char* pv_name;
//  PyObject* py_dbconnection; 
//  if(!PyArg_ParseTuple(args, "sO", &pv_name, &py_dbconnection))
//  {
//    std::cerr << "get_db_epics needs a pv and a db connection as args!" << std::endl;
//    return NULL;
//  }
//  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
//  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
//  { 
//    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
//    std::string sql;
//    for(int dbs_indx = 0; dbs_indx < (dbconn->dbs).size(); ++dbs_indx)
//    {
//      sql = "select value from " + (dbconn->dbs)[dbs_indx] +
//             ".epics_channel where lcs_name = '" + std::string(pv_name) + "'";
////      std::cout << sql << std::endl;
//      StringArray value_trpl = GetDataArrayFromDB(dbconn->db_conn, sql.c_str());
//      if(value_trpl.empty()) 
//        continue;
//      else
//      {
//        std::string value_str;
////        if(value_trpl[0].first == "delay" || value_trpl[0].first == "buncher_on_off")
////          value_str = value_trpl[0].second.second;
////        else
//        value_str = value_trpl[0];
//        return PyString_FromString(value_str.c_str());
//      }
//    }// for dbs_indx
//    std::cerr << "Error in get_db_epics : can't find pv: " << pv_name << ", or this record has a NULL value " << std::endl;  
//  }// if PyObject_IsInstance
//  Py_INCREF(Py_None);
//  return Py_None;
//}
//
//PyDoc_STRVAR(get_db_model__doc__,
//"get_db_model(record_name, field_name, DBConnection) ->\n\n"
//"Get a model parameter value from the db."
//);
//static PyObject* GetDbModel(PyObject* self, PyObject* args)
//{
//  char* record_name, *field_name;
//  PyObject* py_dbconnection; 
//  if(!PyArg_ParseTuple(args, "ssO", &record_name, &field_name, &py_dbconnection))
//  {
//    std::cerr << "get_db_model needs a record_name, a field_name, and a db "
//      "connection as args!" << std::endl;
//    return NULL;
//  }
//  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
//  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
//  { 
//    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
//    std::string sql;
//    for(int dbs_indx = 0; dbs_indx < (dbconn->dbs).size(); ++dbs_indx)
//    {
//      sql = "select name from " + (dbconn->dbs)[dbs_indx] + ".sqlite_master " +
//            "where type = 'table'";  
//      StringArray tbl_names = GetDataArrayFromDB(dbconn->db_conn, sql.c_str()); 
//      int saindx = 0;
//      while(saindx < tbl_names.size())
//      {
//        if(tbl_names[saindx] == "epics_channel")
//        {
//          saindx++;
//          continue;
//        }
//        sql = "select * from " + (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx];
//        StringArray col_names = GetTableColumnNames(dbconn->db_conn, sql.c_str());
//        StringArray::iterator it = std::find(col_names.begin(), col_names.end(), 
//                                              std::string(field_name));
//        if(it != col_names.end())
//        {
//          sql = "select " + std::string(field_name) + " from " + 
//            (dbconn->dbs)[dbs_indx] + "." + tbl_names[saindx] + 
//            " where name = '" + std::string(record_name) + "'";
//          std::string value_str = GetDataFromDB(dbconn->db_conn, sql.c_str());
//          if(value_str != "")
//          {
////            double value = std::atof(value_str.c_str());
////            return PyFloat_FromDouble(value);
//            return PyString_FromString(value_str.c_str());            
//          }
//          else
//            saindx++;
//        }
//        else
//          saindx++;
//      } // while
//    }// for dbs_indx
//    std::cerr << "Error in get_db_model: can't find record name: " << record_name 
//      << " or its field name: " << field_name << std::endl;  
//  }// if PyObject_IsInstance
//  Py_INCREF(Py_None);
//  return Py_None;
//}
//
//PyDoc_STRVAR(get_element_list__doc__, 
//"get_element_list(db = DBConnection, start = start_element (optional), end = end_element (optional), type = element_type (optional)) ->\n\n"
//"Get a list of element names in the range of [start, end] (inclusive)."
//);
//static PyObject* GetElementList(PyObject* self, PyObject* args, PyObject* kwds)
//{
//  char* start_name = "", *end_name = "", *type = "";
//  PyObject* py_dbconnection; 
//  static char *kwlist[] = {"db", "start", "end", "type", NULL};
//  if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|sss", kwlist, &py_dbconnection, &start_name, &end_name, &type))
//  {
//    std::cerr << "get_element_list needs at least a db connection as its arg. "
//      "optional(start element name, end element name, element type)" << std::endl;
//    return NULL;
//  }
//  PyObject* py_dbconnection_type = getHPSimType("DBConnection");
//  if(PyObject_IsInstance(py_dbconnection, py_dbconnection_type))
//  { 
//    DBConnection* dbconn = (DBConnection*)((CPPClassObject*)py_dbconnection)->cpp_obj;
//    int start_indx = 0, end_indx = 0;
//    std::string sql, start_indx_str, end_indx_str;
//    
//    int dbs_indx = 0;
//    std::string model_index = "";
//
//    // get start element model_index
//    if(start_name == "")
//    {
//      start_indx = 0;
//      std::ostringstream oss;
//      oss << start_indx;
//      start_indx_str = oss.str();
//    }
//    else
//    {
//      while(dbs_indx < (dbconn->dbs).size() && model_index == "")
//      {
//        sql = "select model_index from " + (dbconn->dbs)[dbs_indx] +
//          ".linac where name = '" + std::string(start_name) + "'";
//         model_index = GetDataFromDB(dbconn->db_conn, sql.c_str());
//         ++dbs_indx;
//      }
//      if(model_index == "")
//      {
//        std::cerr << "The start element: " << std::string(start_name) << " doesn't have a model_index!"  << std::endl;
//        Py_INCREF(Py_None);
//        return Py_None;
//      }
//      else
//      {
//        start_indx_str = model_index;
//        start_indx = std::atoi(model_index.c_str());
//      }
//    }
//    // get end element model_index
//    dbs_indx = 0;
//    model_index = "";
//    if(end_name == "")
//    {
//      sql = "select max(model_index) from " + 
//            (dbconn->dbs)[(dbconn->dbs).size() - 1] + ".linac";
//      model_index = GetDataFromDB(dbconn->db_conn, sql.c_str());
//      end_indx_str = model_index;
//      end_indx = std::atoi(end_indx_str.c_str());
//    }
//    else
//    {
//      while(dbs_indx < (dbconn->dbs).size() && model_index == "")
//      {
//        sql = "select model_index from " + (dbconn->dbs)[dbs_indx] +
//          ".linac where name = '" + std::string(end_name) + "'";
//         model_index = GetDataFromDB(dbconn->db_conn, sql.c_str());
//         ++dbs_indx;
//      }
//      if(model_index == "")
//      {
//        std::cerr << "The end element: " << std::string(end_name) << " doesn't have a model_index!"  << std::endl;
//        Py_INCREF(Py_None);
//        return Py_None;
//      }
//      else
//      {
//        end_indx_str = model_index;
//        end_indx = std::atoi(model_index.c_str());
//      }
//    }
//
//    if(end_indx < start_indx)
//    {
//      std::cerr << "The model_index of the downstream element [" << end_name 
//        << ", " << end_indx << "] is smaller than that of the upstream element [" 
//        << start_name << ", " << start_indx << "]" << std::endl;
//      Py_INCREF(Py_None);
//      return Py_None;
//    }   
//
//    std::vector<std::string> final_list;
//    dbs_indx = 0;
//    while(dbs_indx < (dbconn->dbs).size())
//    {
//      sql = "select name, model_type from " + (dbconn->dbs)[dbs_indx] +
//          ".linac where model_index >= " + start_indx_str + 
//          " and model_index <= " + end_indx_str;
//      if(type != "")
//        sql = sql + " and model_type = '" + type + "'";
//      StringPairArray nt_list = GetDataPairArrayFromDB(dbconn->db_conn, sql.c_str());
//      if(!nt_list.empty())
//      {
//        for(int i = 0; i < nt_list.size(); ++i)
//          final_list.push_back(nt_list[i].first);
//      }
//      ++dbs_indx;
//    }
//    if(!final_list.empty())
//    {
//      PyObject* elem_lst = PyList_New(final_list.size());
//      for(int i = 0; i < final_list.size(); ++i)
//        PyList_SetItem(elem_lst, i, PyString_FromString(final_list[i].c_str()));
//      return elem_lst;
//    }
//
////    PyObject* elem_lst = PyList_New(end_indx - start_indx + 1);
////    int cnt = 0;
////    for(int i = start_indx; i <= end_indx; ++i)
////    {
////      std::ostringstream osstr;
////      osstr << i;
////      dbs_indx = 0;
////      std::string elem_name = "";
////      std::string elem_type = "";
////      while(dbs_indx < (dbconn->dbs).size() && elem_name == "")
////      {
////        sql = "select name, model_type from " + (dbconn->dbs)[dbs_indx] +
////          ".linac where model_index = " + osstr.str();
////        StringPairArray view_info = GetDataPairArrayFromDB(dbconn->db_conn, sql.c_str());
////        if(!view_info.empty())
////        {
////          std::cout << "Found!" << std::endl;
////          elem_name = view_info[0].first;
////          elem_type = view_info[0].second;
////        }
////        ++dbs_indx;
////      }
////      if(elem_name == "")
////      {
////        std::cerr << "model_index : " << osstr.str() << " doesn't have a name!"  << std::endl;
////        Py_INCREF(Py_None);
////        return Py_None;
////      }
////      else if(elem_name != "" /*&& elem_type == type*/)
////      {
////        PyObject* aname = PyString_FromString(elem_name.c_str());
////        PyList_SetItem(elem_lst, cnt, aname);
////        ++cnt;
////      }// if elem_name 
////    }// for i 
//  }// if PyObject_IsInstance
//  Py_INCREF(Py_None);
//  return Py_None;
//}
//
//static PyMethodDef HPSimModuleMethods[]={
//  {"set_gpu", (PyCFunction)SetGPU, METH_VARARGS, set_gpu__doc__}, 
//  {"set_db_epics", (PyCFunction)SetDbEPICS, METH_VARARGS, set_db_epics__doc__}, 
//  {"set_db_model", (PyCFunction)SetDbModel, METH_VARARGS, set_db_model__doc__}, 
//  {"get_db_epics", (PyCFunction)GetDbEPICS, METH_VARARGS, get_db_epics__doc__}, 
//  {"get_db_model", (PyCFunction)GetDbModel, METH_VARARGS, get_db_model__doc__}, 
//  {"get_element_list", (PyCFunction)GetElementList, METH_VARARGS|METH_KEYWORDS, get_element_list__doc__}, 
//  {NULL}
//};

static PyMethodDef HPSimModuleMethods[]={
  {NULL}
};

PyMODINIT_FUNC initHPSim()
{
  PyObject* module = Py_InitModule("HPSim", HPSimModuleMethods);
//  import_array();
  initBeam(module);
//  initDBConnection(module);
//  initBeamLine(module);
//  initSimulator(module);
//  initSpaceCharge(module);
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
