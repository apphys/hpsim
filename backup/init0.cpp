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

void GenerateBeamLine(std::vector<BeamLineElement>& r_linac, DBConnection* r_db_conn)
{
  std::vector<std::string> dbs = r_db_conn->dbs;
  sqlite3* db_conn = r_db_conn->db_conn;
  char* errmsg;
  std::ostringstream sstr;
  std::ostringstream msstr;
  uint model_index = 0;
  std::string view_index;
  std::string tank_index;
  std::string sql, tmp = "";
  sqlite3_exec(db_conn, "BEGIN TRANSACTION", NULL, NULL, &errmsg);

  for(int dbs_indx = 0; dbs_indx < dbs.size(); ++dbs_indx)
  {
    // start a new beamline db 
    sql = "select view_index, model_type from " + dbs[dbs_indx] + ".linac";
    std::vector<std::pair<std::string, std::string> > elems = GetDataPairArrayFromDB(db_conn, sql.c_str());
    for(int elem_indx = 0; elem_indx < elems.size(); ++elem_indx)
    {
      view_index = elems[elem_indx].first;
      tmp = elems[elem_indx].second;
      BeamLineElement elem;
      if(tmp == "quad")
      {
        elem.type[0] = 'q';
        elem.type[1] = 'd';
        sql = "select gradient_model from " + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.gradient = std::atof(tmp.c_str());
        sql = "select length_model from "  + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.length = std::atof(tmp.c_str());
        sql = "select aperture_model from "  + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.aperture1 = std::atof(tmp.c_str());
        sql = "select monitor from "  + dbs[dbs_indx] + ".quad where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atoi(tmp.c_str());
        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".quad set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec quad: "+view_index);
        tmp = "";
      }
      else if (tmp == "steerer") 
      {
        elem.type[0] = 's';
        elem.type[1] = 't';
        sql = "select bl_h_model from " + dbs[dbs_indx] + ".steerer where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.tp = std::atof(tmp.c_str());
        sql = "select bl_v_model from " + dbs[dbs_indx] + ".steerer where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.sp = std::atof(tmp.c_str());
        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".steerer set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec steerer: "+view_index);
        tmp = "";
      }
      else if(tmp == "dtl_gap" || tmp == "ccl_gap")
      {
        elem.type[0] = 'g';
        // !!!!!!!!!!!!!!! need to change exactly the same!
        if (tmp == "dtl_gap")
        {
          elem.type[1] = 'd';
          sql = "select t1.beta_g, t1.beta_min, t1.ta0, t1.ta1, t1.ta2, t1.ta3, t1.ta4, t1.ta5, t1.sa1, "
                "t1.sa2, t1.sa3, t1.sa4, t1.sa5 from " + dbs[dbs_indx] + ".transit_time_factor t1 join " + 
                dbs[dbs_indx] + ".rf_gap t2 on t1.module_id = t2.module_id and t1.tank_id = t2.cell_id "
                "where t2.view_index = " + view_index;
          std::vector<std::vector<std::string> > querydata = GetQueryResults(db_conn, sql.c_str());
          elem.sf_betag = std::atof(querydata[0][0].c_str());
          elem.sf_betamin = std::atof(querydata[0][1].c_str());
          elem.sf_t_a0 = std::atof(querydata[0][2].c_str());
          elem.sf_t_a1 = std::atof(querydata[0][3].c_str());
          elem.sf_t_a2 = std::atof(querydata[0][4].c_str());
          elem.sf_t_a3 = std::atof(querydata[0][5].c_str());
          elem.sf_t_a4 = std::atof(querydata[0][6].c_str());
          elem.sf_t_a5 = std::atof(querydata[0][7].c_str());
          elem.sf_s_a1 = std::atof(querydata[0][8].c_str());
          elem.sf_s_a2 = std::atof(querydata[0][9].c_str());
          elem.sf_s_a3 = std::atof(querydata[0][10].c_str());
          elem.sf_s_a4 = std::atof(querydata[0][11].c_str());
          elem.sf_s_a5 = std::atof(querydata[0][12].c_str());
        }
        else if (tmp == "ccl_gap")
        {
          elem.type[1] = 'c';
          sql = "select t1.beta_g, t1.beta_min, t1.ta0, t1.ta1, t1.ta2, t1.ta3, t1.ta4, t1.ta5, t1.sa1, "
                "t1.sa2, t1.sa3, t1.sa4, t1.sa5 from " + dbs[dbs_indx] + ".transit_time_factor t1 join " + 
                dbs[dbs_indx] + ".rf_gap t2 on t1.module_id = t2.module_id and t1.tank_id = t2.tank_id "
                "where t2.view_index = " + view_index;
          std::vector<std::vector<std::string> > querydata = GetQueryResults(db_conn, sql.c_str());
          elem.sf_betag = std::atof(querydata[0][0].c_str());
          elem.sf_betamin = std::atof(querydata[0][1].c_str());
          elem.sf_t_a0 = std::atof(querydata[0][2].c_str());
          elem.sf_t_a1 = std::atof(querydata[0][3].c_str());
          elem.sf_t_a2 = std::atof(querydata[0][4].c_str());
          elem.sf_t_a3 = std::atof(querydata[0][5].c_str());
          elem.sf_t_a4 = std::atof(querydata[0][6].c_str());
          elem.sf_t_a5 = std::atof(querydata[0][7].c_str());
          elem.sf_s_a1 = std::atof(querydata[0][8].c_str());
          elem.sf_s_a2 = std::atof(querydata[0][9].c_str());
          elem.sf_s_a3 = std::atof(querydata[0][10].c_str());
          elem.sf_s_a4 = std::atof(querydata[0][11].c_str());
          elem.sf_s_a5 = std::atof(querydata[0][12].c_str());
        }
        else 
        {
          std::cerr << "Error from init.cpp::GenerateBeamLine(): invalid rf_gap type! "
            << "Options are: 'dtl_gap' or 'ccl_gap'." << std::endl;
          exit(0);
        }
        sql = "select length_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.length = std::atof(tmp.c_str());
        sql = "select frequency_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.freq = std::atof(tmp.c_str());
        sql = "select cell_length_betalambda_design from " + dbs[dbs_indx] + ".rf_module where id = (select module_id from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index+")";
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.cell_len_bl = std::atof(tmp.c_str());
        sql = "select amplitude_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.rf_amp = std::atof(tmp.c_str());
        sql = "select ref_phase_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.phi_c = std::atof(tmp.c_str());
        sql = "select beam_phase_shift_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.phi_out = std::atof(tmp.c_str());
        sql = "select energy_out_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.energy_out = std::atof(tmp.c_str());
        sql = "select beta_center_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.beta_g = std::atof(tmp.c_str());
        sql = "select dg_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.dg = std::atof(tmp.c_str());
        sql = "select t_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atof(tmp.c_str());
        sql = "select tp_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.tp = std::atof(tmp.c_str());
        sql = "select sp_model from " + dbs[dbs_indx] + ".rf_gap where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.sp = std::atof(tmp.c_str());
        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".rf_gap set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec gap: " + view_index);
        tmp = "";
      }
      else if(tmp == "diagnostics")
      {
        elem.type[0] = 'd';
        elem.type[1] = 'g';
        sql = "select monitor from "  + dbs[dbs_indx] + ".diagnostics where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atoi(tmp.c_str());
        elem.length = 0.0;

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".diagnostics set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec drift: " + view_index);
        tmp = "";
      }
      else if(tmp == "drift")
      {
        elem.type[0] = 'd';
        elem.type[1] = 'r';
        sql = "select length_model from " + dbs[dbs_indx] + ".drift where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.length = std::atof(tmp.c_str());

        sql = "select aperture_model from " + dbs[dbs_indx] + ".drift where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.aperture1 = std::atof(tmp.c_str());

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".drift set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec drift: " + view_index);
        tmp = "";
      }
      else if(tmp == "dipole")
      {
        elem.type[0] = 'd';
        elem.type[1] = 'p';
        sql = "select rho_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.length = std::atof(tmp.c_str());

        sql = "select angle_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.gradient = std::atof(tmp.c_str());

        sql = "select edge_angle1_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.tp = std::atof(tmp.c_str());

        sql = "select edge_angle2_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.sp = std::atof(tmp.c_str());

        sql = "select half_gap_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.dg = std::atof(tmp.c_str());

        sql = "select k1_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.phi_c = std::atof(tmp.c_str());

        sql = "select k2_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.phi_out = std::atof(tmp.c_str());

        sql = "select field_index_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atof(tmp.c_str());

        sql = "select kenergy_model from " + dbs[dbs_indx] + ".dipole where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.energy_out = std::atof(tmp.c_str());

        elem.aperture1 = 0.0;
        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".dipole set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec dipole: " + view_index);
        tmp = "";
      }
      else if(tmp == "buncher")
      {
        elem.type[0] = 'b';
        elem.type[1] = 'c';
        sql = "select voltage_model from " + dbs[dbs_indx] + ".buncher where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.rf_amp = std::atof(tmp.c_str());

        sql = "select frequency_model from " + dbs[dbs_indx] + ".buncher where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.freq = std::atof(tmp.c_str());

        sql = "select phase_model from " + dbs[dbs_indx] + ".buncher where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.phi_c = std::atof(tmp.c_str());

        sql = "select aperture_model from " + dbs[dbs_indx] + ".buncher where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.aperture1= std::atof(tmp.c_str());

        sql = "select on_off from " + dbs[dbs_indx] + ".buncher where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atof(tmp.c_str());

        elem.length = 0.0;

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".buncher set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec buncher: " + view_index);
        tmp = "";
      }
      else if(tmp == "rotation")
      {
        elem.type[0] = 'r';
        elem.type[1] = 'o';
        sql = "select angle_model from " + dbs[dbs_indx] + ".rotation where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.phi_c = std::atof(tmp.c_str());
        elem.length = 0.0;

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".rotation set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec rotation: " + view_index);
        tmp = "";
      }
      else if(tmp == "caperture")
      {
        elem.type[0] = 'c';
        elem.type[1] = 'a';
        sql = "select aperture_model from " + dbs[dbs_indx] + ".caperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.aperture1 = std::atof(tmp.c_str());

        sql = "select in_out_model from " + dbs[dbs_indx] + ".caperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atof(tmp.c_str());
        elem.length = 0.0;

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".caperture set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec caperture: " + view_index);
        tmp = "";
      }   
      else if(tmp == "raperture")
      {
        elem.type[0] = 'r';
        elem.type[1] = 'a';
        sql = "select aperture_xl_model from " + dbs[dbs_indx] + ".raperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.aperture1 = std::atof(tmp.c_str());
        sql = "select aperture_xr_model from " + dbs[dbs_indx] + ".raperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.tp = std::atof(tmp.c_str());
        sql = "select aperture_yt_model from " + dbs[dbs_indx] + ".raperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.aperture2 = std::atof(tmp.c_str());
        sql = "select aperture_yb_model from " + dbs[dbs_indx] + ".raperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.sp = std::atof(tmp.c_str());
        sql = "select in_out_model from " + dbs[dbs_indx] + ".raperture where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atof(tmp.c_str());
        elem.length = 0.0;

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".raperture set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec raperture: " + view_index);
        tmp = "";
      }   
      else if(tmp == "spch_comp")
      {
        elem.type[0] = 's';
        elem.type[1] = 'c';
        sql = "select fraction_model from " + dbs[dbs_indx] + ".spch_comp where view_index = " + view_index;
        tmp = GetDataFromDB(db_conn, sql.c_str());
        elem.t = std::atof(tmp.c_str());
        elem.length = 0.0;

        r_linac.push_back(elem);

        msstr.str("");
        msstr << model_index++;
        sql = "update " + dbs[dbs_indx] + ".spch_comp set model_index = " + msstr.str() + " where view_index = " + view_index;
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), "sqlite3_exec spch: " + view_index);
        tmp = "";
      }   
    }
  } // for
  sqlite3_exec(db_conn, "END TRANSACTION", NULL, NULL, &errmsg);
}

