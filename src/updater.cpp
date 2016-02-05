#include <sstream>
#include <iostream>
#include <cstdlib>
#include "updater.h"
#include "sql_utility.h"
#include "tool_lib.h"
#ifdef _DEBUG
  #include <iostream>
#endif

/*!
 * \note The reason the constructor does not take a db and a beamline pointer
 * as inputs is so that when this is constructed, there is no need for a db
 * and a beamline to be present. But this class does need both of them
 * in order to run its functions so one needs to follow the steps to set up
 * the db and beamline correctly before one can start the updates
 */
Updater::Updater(std::string r_pv) 
  : pv_(r_pv), val_(""), old_val_(""), db_conn_(NULL), beamline_ptr_(NULL)
{
}
/*!
 * \note The db has to be opened and external library has to be loaded in sqlite3 
 * in the server thread before this method can be called. 
 */
void Updater::UpdateDB()
{
  char* errmsg;
  sqlite3_exec(db_conn_, "BEGIN TRANSACTION", NULL, NULL, &errmsg);
  std::string sql = "update " + db_ + ".epics_channel set value = " + val_ + " where lcs_name = '" + pv_ + "'";
  sqlite3_stmt* stmt_;
  SQLCheck(sqlite3_prepare_v2(db_conn_, sql.c_str(), -1, &stmt_, NULL), "sqlite3_prepare: " + sql);
  sqlite3_step(stmt_);
  SQLCheck(sqlite3_finalize(stmt_), stmt_, "sqlite3_finalize for PV:" + pv_);
  sqlite3_exec(db_conn_, "END TRANSACTION", NULL, NULL, &errmsg);
}
                                   
void Updater::UpdateModel()
{
  typedef std::vector<std::pair<std::string, std::string> > StringPairArray;
  typedef std::vector<std::pair<std::string, std::pair<std::string, std::string> > > StringTripletArray;

  char* errmsg;  
  sqlite3_exec(db_conn_, "BEGIN TRANSACTION", NULL, NULL, &errmsg);
//  if(pv_ == "MRPH001D01")
//  {
//    std::string sql = "select model_index, beam_phase_shift_model from " + db_ +
//      ".rf_gap where frequency_model = 805";
//    StringPairArray gap_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
//    for(int j = 0; j < gap_info.size(); ++j)
//    {
//      int index = std::atoi(gap_info[j].first.c_str());
//      RFGap* gp = dynamic_cast<RFGap*>(beamline_ptr_[index]);
//      gp->SetPhaseShift(std::atof(gap_info[j].second.c_str()));
//    }
//    sqlite3_exec(db_conn_, "END TRANSACTION", NULL, NULL, &errmsg);
//    return;
//  } 
  std::string sql = "select model_type, name from " + db_ + ".channel_list where channel1='" + 
      pv_ + "' or channel2='" + pv_ + "' or channel3 ='" + pv_ + "'";
  StringPairArray pv_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
  if(pv_info.size() > 0)
  {
    for(int i = 0; i < pv_info.size(); ++i)
    {
      std::string type = pv_info[i].first;
      std::string name = pv_info[i].second;
      // raperture
//      if(type == "raperture")
//      {
//        sql = "select model_index, aperture_xl_model, aperture_xr_model from " + db_ + ".raperture where name = '" + name + "'";
//        StringTripletArray raper_info = GetDataTripletArrayFromDB(db_conn_, sql.c_str());
//        if(!raper_info.empty())
//        {
//          int index = std::atoi(raper_info[0].first.c_str());
//          beamline_ptr_[index].aperture1 = std::atof(raper_info[0].second.first.c_str());
//          beamline_ptr_[index].tp = std::atof(raper_info[0].second.second.c_str());
//        }
//        else
//          std::cerr << "Updater::UpdateModel(), no model_index & aperture_xl_model & aperture_xr_model were found for "
//            "raperture with name: " << name << std::endl;
//      }
//      // dipole
//      if(type == "dipole")
//      {
//        sql = "select model_index, rho_model, kenergy_model from " + db_ + ".dipole where name = '" + name + "'";
//        StringTripletArray dipole_info = GetDataTripletArrayFromDB(db_conn_, sql.c_str());
//        if(!dipole_info.empty())
//        {
//          int index = std::atoi(dipole_info[0].first.c_str());
//          beamline_ptr_[index].length = std::atof(dipole_info[0].second.first.c_str()); // rho
//          beamline_ptr_[index].energy_out = std::atof(dipole_info[0].second.second.c_str()); // rho
//          sql = "select angle_model, edge_angle1_model, edge_angle2_model from " + db_ + 
//                  ".dipole where name = '" + name + "'";
//          StringTripletArray dipole_info2 = GetDataTripletArrayFromDB(db_conn_, sql.c_str());
//          beamline_ptr_[index].gradient = std::atof(dipole_info2[0].first.c_str()); // angle
//          beamline_ptr_[index].tp = std::atof(dipole_info2[0].second.first.c_str()); // edge_angle1
//          beamline_ptr_[index].sp = std::atof(dipole_info2[0].second.second.c_str()); // edge_angle2
//        }
//        else
//          std::cerr << "Updater::UpdateModel(), no model_index & length_model were found for "
//            "dipole with name: " << name << std::endl;
//      }
//      // drift 
//      if(type == "drift")
//      {
//        sql = "select model_index, length_model from " + db_ + ".drift where name = '" + name + "'";
//        StringPairArray drift_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
//        if(!drift_info.empty())
//        {
//          int index = std::atoi(drift_info[0].first.c_str());
//          beamline_ptr_[index].length = std::atof(drift_info[0].second.c_str()); 
//        }
//        else
//         std::cerr << "Updater::UpdateModel(), no model_index & length_model were found for "
//            << "drift with name: " << name << std::endl;
//      }
      // quad
      if(type == "quad")
      {
        sql = "select model_index, gradient_model from " + db_ + ".quad where name = '" + name + "'";
        StringPairArray quad_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
        if(!quad_info.empty())
        {
          int index = std::atoi(quad_info[0].first.c_str());
          Quad* qd  = dynamic_cast<Quad*>((*beamline_ptr_)[index]);
          qd->SetGradient(std::atof(quad_info[0].second.c_str())); //quad
        }
        else
         std::cerr << "Updater::UpdateModel(), no model_index & gradient were found for " 
            << "quad with name: " << name << std::endl; 
      }
      // rf_module
      else if(type == "rf_module")
      {
        sql = "select value_type from " + db_ + ".epics_channel where lcs_name ='" 
              + pv_ + "'";
        std::string pv_type = GetDataFromDB(db_conn_, sql.c_str());
        if(pv_type == "") 
          std::cerr << "No value type is associated with epics channel: " 
            << pv_ << std::endl;
        else if(pv_type == "rf_ph")
        {
          sql = "select model_index, beam_phase_shift_model from " + db_ +
            ".rf_gap where module_id = (select id from " + db_ + ".rf_module "
            "where name = '" + name + "')";
          StringPairArray gap_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
          for(int j = 0; j < gap_info.size(); ++j)
          {
            int index = std::atoi(gap_info[j].first.c_str());
            RFGap* gp = dynamic_cast<RFGap*>((*beamline_ptr_)[index]);
            gp->SetPhaseShift(std::atof(gap_info[j].second.c_str()));
          }
        }
        else if(pv_type == "rf_amp" || pv_type == "delay")
        {
          sql = "select model_index, amplitude_model, ref_phase_model from " 
              + db_ + ".rf_gap where module_id = (select id from " + db_ 
              + ".rf_module where name = '" + name + "')";
          StringTripletArray gap_info = GetDataTripletArrayFromDB(db_conn_, sql.c_str());
          int gap_index; double amp, phi;
          for(int j = 0; j < gap_info.size(); ++j)
          {
            gap_index = std::atoi(gap_info[j].first.c_str());
            amp = std::atof(gap_info[j].second.first.c_str());
            phi = std::atof(gap_info[j].second.second.c_str());
            RFGap* gp = dynamic_cast<RFGap*>((*beamline_ptr_)[gap_index]);
            gp->SetRFAmplitude(amp);
            gp->SetRefPhase(phi);
          }           
        }
        else
          std::cerr << "Wrong type of epics channel is associated with a rf module!" << std::endl;
      }
      // buncher
//      else if(type == "buncher")
//      {
//        sql = "select value_type from " + db_ + ".epics_channel where lcs_name = '" + pv_ + "'";
//        std::string pv_type = GetDataFromDB(db_conn_, sql.c_str());
//        if(pv_type == "") 
//          std::cerr << "No value type is associated with epics channel: " 
//            << pv_ << std::endl;
//        if(pv_type == "buncher_amp")
//        {
//          sql = "select model_index, voltage_model from " + db_  
//            + ".buncher where name = '" + name + "'";
//          StringPairArray buncher_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
//          if(buncher_info[0].second == "")
//          {
//            std::cerr << "voltage_model of the buncher with name = "
//               << name << " can't be empty!"<< std::endl;
//            continue;
//          }
//          int model_index = std::atoi(buncher_info[0].first.c_str());
//          double amp = std::atof(buncher_info[0].second.c_str());
//          beamline_ptr_[model_index].rf_amp= amp;
//        }
//        else if(pv_type == "buncher_ph")
//        {
//          sql = "select model_index, phase_model from " + db_  
//            + ".buncher where name = '" + name + "'";
//          StringPairArray buncher_info= GetDataPairArrayFromDB(db_conn_, sql.c_str());
//          if(buncher_info[0].second == "")
//          {
//            std::cerr << "phase_model of the buncher with name = "
//               << name << " can't be empty!"<< std::endl;
//            continue;
//          }
//          int model_index = std::atoi(buncher_info[0].first.c_str());
//          double phase = std::atof(buncher_info[0].second.c_str());
//          beamline_ptr_[model_index].phi_c = phase;
//        }
//        else if(pv_type == "buncher_on_off")
//        {
//          sql = "select model_index, on_off from " + db_  
//            + ".buncher  where name = '" + name + "'";
//          StringPairArray buncher_info = GetDataPairArrayFromDB(db_conn_, sql.c_str());
//          if(buncher_info[0].second == "")
//          {
//            std::cerr << "on_off of the buncher with name = "
//               << name << " can't be empty!"<< std::endl;
//            continue;
//          }
//          int model_index = std::atoi(buncher_info[0].first.c_str());
//          double onoff = std::atof(buncher_info[0].second.c_str());
//          beamline_ptr_[model_index].t = onoff;
//        }// if pv_type
//        else 
//          std::cerr << "Wrong type of epics channel is associated with a buncher!" << std::endl;
//      } // if type 
    } //for
  } // if pv_info.size
  else
    std::cerr << "PV name: " << pv_ << " is not used by any element in " + db_ + "!" << std::endl;

  sqlite3_exec(db_conn_, "END TRANSACTION", NULL, NULL, &errmsg);
}

