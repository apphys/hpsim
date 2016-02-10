#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <sstream>
#include <cstdlib>
#include <vector>
#include "init.h"
#include "sql_utility.h"


void SetGPU(int r_id)
{
  cudaSetDevice(r_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, r_id);
  std::cout << "GPU : " << props.name << std::endl;
}

void GenerateBeamLine(BeamLine& r_linac, DBConnection* r_db_conn)
{
  std::vector<std::string> dbs = r_db_conn->dbs;
  sqlite3* db_conn = r_db_conn->db_conn;
  char* errmsg;
  std::ostringstream sstr;
  std::ostringstream msstr;
  uint model_index = 0;
  std::string name; 
  std::string view_index;
  std::string tank_index;
  std::string sql, tmp = "";
  sqlite3_exec(db_conn, "BEGIN TRANSACTION", NULL, NULL, &errmsg);

  for(int dbs_indx = 0; dbs_indx < dbs.size(); ++dbs_indx)
  {
    // start a new beamline db 
    sql = "select name, view_index, model_type from " + dbs[dbs_indx] + ".linac";
    std::vector<std::pair<std::string, std::pair<std::string, std::string> > > elems = GetDataTripletArrayFromDB(db_conn, sql.c_str());
    for(int elem_indx = 0; elem_indx < elems.size(); ++elem_indx)
    {
      name = elems[elem_indx].first;
      view_index = elems[elem_indx].second.first;
      tmp = elems[elem_indx].second.second;

      if(tmp == "quad")
      {
        Quad* elem = new Quad(name);
        sql = "select gradient_model from " + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetGradient(std::atof(tmp.c_str()));
        sql = "select length_model from "  + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetLength(std::atof(tmp.c_str()));
        sql = "select aperture_model from "  + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetAperture(std::atof(tmp.c_str()));
        sql = "select monitor from "  + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        if(std::atoi(tmp.c_str()) > 0)
          elem->SetMonitorOn();
        r_linac.AddElement(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".quad set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec quad: "+view_index);
        tmp = "";
      }
      else if(tmp == "drift")
      {
        Drift* elem = new Drift(name);
        sql = "select length_model from " + dbs[dbs_indx] + ".drift where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetLength(std::atof(tmp.c_str()));

        sql = "select aperture_model from " + dbs[dbs_indx] + ".drift where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetAperture(std::atof(tmp.c_str()));

        r_linac.AddElement(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".drift set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec drift: " + view_index);
        tmp = "";
      }
      else if(tmp == "dtl_gap" || tmp == "ccl_gap")
      {
        RFGap* elem = new RFGap(name, "RFGap-DTL");
        if (tmp == "ccl_gap")
          elem->SetType("RFGap-CCL");

        if(tmp == "dtl_gap")
          sql = "select t1.beta_g, t1.beta_min, t1.ta0, t1.ta1, t1.ta2, t1.ta3, t1.ta4, t1.ta5, t1.sa1, "
                "t1.sa2, t1.sa3, t1.sa4, t1.sa5 from " + dbs[dbs_indx] + ".transit_time_factor t1 join " + 
                dbs[dbs_indx] + ".rf_gap t2 on t1.module_id = t2.module_id and t1.tank_id = t2.cell_id "
                "where t2.view_index = " + view_index;
        else // ccl_gap
          sql = "select t1.beta_g, t1.beta_min, t1.ta0, t1.ta1, t1.ta2, t1.ta3, t1.ta4, t1.ta5, t1.sa1, "
                "t1.sa2, t1.sa3, t1.sa4, t1.sa5 from " + dbs[dbs_indx] + ".transit_time_factor t1 join " + 
                dbs[dbs_indx] + ".rf_gap t2 on t1.module_id = t2.module_id and t1.tank_id = t2.tank_id "
                "where t2.view_index = " + view_index;

        std::vector<std::vector<std::string> > querydata = GetQueryResults(db_conn, sql.c_str());
        elem->SetFitBetaCenter(std::atof(querydata[0][0].c_str()));
        elem->SetFitBetaMin(std::atof(querydata[0][1].c_str()));
        elem->SetFitT0(std::atof(querydata[0][2].c_str()));
        elem->SetFitT1(std::atof(querydata[0][3].c_str()));
        elem->SetFitT2(std::atof(querydata[0][4].c_str()));
        elem->SetFitT3(std::atof(querydata[0][5].c_str()));
        elem->SetFitT4(std::atof(querydata[0][6].c_str()));
        elem->SetFitT5(std::atof(querydata[0][7].c_str()));
        elem->SetFitS1(std::atof(querydata[0][8].c_str()));
        elem->SetFitS2(std::atof(querydata[0][9].c_str()));
        elem->SetFitS3(std::atof(querydata[0][10].c_str()));
        elem->SetFitS4(std::atof(querydata[0][11].c_str()));
        elem->SetFitS5(std::atof(querydata[0][12].c_str()));

        sql = "select length_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetLength(std::atof(tmp.c_str()));
        sql = "select frequency_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetFrequency(std::atof(tmp.c_str()));
        sql = "select cell_length_betalambda_design from " + dbs[dbs_indx] + ".rf_module where id = (select module_id from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index+")";
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetCellLengthOverBetaLambda(std::atof(tmp.c_str()));
        sql = "select amplitude_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetRFAmplitude(std::atof(tmp.c_str()));
        sql = "select ref_phase_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetRefPhase(std::atof(tmp.c_str()));
        sql = "select beam_phase_shift_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetPhaseShift(std::atof(tmp.c_str()));
        sql = "select energy_out_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetEnergyOut(std::atof(tmp.c_str()));
        sql = "select beta_center_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetBetaCenter(std::atof(tmp.c_str()));
        sql = "select dg_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetDg(std::atof(tmp.c_str()));
        sql = "select t_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetT(std::atof(tmp.c_str()));
        sql = "select tp_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetTp(std::atof(tmp.c_str()));
        sql = "select sp_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetSp(std::atof(tmp.c_str()));

        r_linac.AddElement(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".rf_gap set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec gap: " + view_index);
        tmp = "";
      }
      else if(tmp == "rotation")
      {
        Rotation* elem = new Rotation(name);
        sql = "select angle_model from " + dbs[dbs_indx] + ".rotation where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem->SetAngle(std::atof(tmp.c_str()));

        r_linac.AddElement(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".rotation set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec rotation: " + view_index);
        tmp = "";
      }
    }
  } // for
  sqlite3_exec(db_conn, "END TRANSACTION", NULL, NULL, &errmsg);
}

