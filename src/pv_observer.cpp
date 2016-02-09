#include <cstdlib>
#include <iostream>
#include "pv_observer.h"
#include "sql_utility.h"

PVObserver::PVObserver(std::string r_pv, std::string r_db): pv_(r_pv), db_(r_db)
{
}
void PVObserver::Update(std::string r_val)
{
  val_ = r_val;
  UpdateDB();
  UpdateModel();
}
void PVObserver::UpdateDB() 
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

QuadPVObserver::QuadPVObserver(std::string r_pv, std::string r_db) 
  : PVObserver(r_pv, r_db)
{
}

void QuadPVObserver::AttachBeamLineElement(BeamLineElement* r_elem)
{
  if (Quad* tmp_quad = dynamic_cast<Quad*>(r_elem))
    quad_.push_back(tmp_quad);
  else
  {
    std::cerr << "Cann't attach " << r_elem->GetName() 
      << " to a QuadPVObserver!" << std::endl;
    exit(-1);
  }
}

void QuadPVObserver::UpdateModel()
{
//  char* errmsg;  
//  sqlite3_exec(GetDBconn(), "BEGIN TRANSACTION", NULL, NULL, &errmsg);

  for(int i = 0; i < quad_.size(); ++i)
  {
    std::string sql = "select gradient_model from " + GetDB() + ".quad where name = '" + quad_[i]->GetName() + "'";
    std::string data = GetDataFromDB(GetDBconn(), sql.c_str());
    if (data != "")
      quad_[i]->SetGradient(std::atof(data.c_str()));
    else
     std::cerr << "QuadPVObserver::UpdateModel() failed, no gradient were found for " 
        << "quad : " << quad_[i]->GetName() << std::endl; 
  }
//sqlite3_exec(GetDBconn(), "END TRANSACTION", NULL, NULL, &errmsg);
}

std::vector<std::string> QuadPVObserver::GetElementNames() const
{
  std::vector<std::string> rlt(quad_.size(), "");
  for(int i = 0; i < quad_.size(); ++i)
    rlt[i] = quad_[i]->GetName(); 
  return rlt; 
}

RFPhasePVObserver::RFPhasePVObserver(std::string r_pv, std::string r_db)
  : PVObserver(r_pv, r_db)
{
}

void RFPhasePVObserver::UpdateModel()
{
  for(int i = 0; i < gap_.size(); ++i)
  {
    std::string sql = "select beam_phase_shift_model from " + GetDB() + ".rf_gap where name = '" + gap_[i]->GetName() + "'";
    std::string data = GetDataFromDB(GetDBconn(), sql.c_str());
    if(data != "")
      gap_[i]->SetPhaseShift(std::atof(data.c_str()));
    else
      std::cerr << "RFPhasePVObserver::UpdateModel() failed, no beam_phase_shift"
        "were found for RFGap: " << gap_[i]->GetName() << std::endl;
  }
}

void RFPhasePVObserver::AttachBeamLineElement(BeamLineElement* r_elem)
{
  if (RFGap* tmp_gap = dynamic_cast<RFGap*>(r_elem))
    gap_.push_back(tmp_gap);
  else
  {
    std::cerr << "Cann't attach " << r_elem->GetName() 
      << " to a RFPhasePVObserver!" << std::endl;
    exit(-1);
  }
}

std::vector<std::string> RFPhasePVObserver::GetElementNames() const
{
  std::vector<std::string> rlt(gap_.size(), "");
  for(int i = 0; i < gap_.size(); ++i)
    rlt[i] = gap_[i]->GetName(); 
  return rlt; 
}
RFAmplitudePVObserver::RFAmplitudePVObserver(std::string r_pv, std::string r_db)
  : PVObserver(r_pv, r_db)
{
}

void RFAmplitudePVObserver::UpdateModel()
{
  for(int i = 0; i < gap_.size(); ++i)
  {
    std::string sql = "select amplitude_model, ref_phase_model from " 
                + GetDB() + ".rf_gap where name = '" + gap_[i]->GetName() + "'";
    std::vector<std::vector<std::string> > data = GetQueryResults(GetDBconn(), sql.c_str());
    if(!data.empty())
    {
      gap_[i]->SetRFAmplitude(std::atof(data[0][0].c_str()));
      gap_[i]->SetRefPhase(std::atof(data[0][1].c_str()));
    }
    else
      std::cerr << "RFAmplitudePVObserver::UpdateModel() failed, no amplitude & ref_phase "
        "were found for RFGap: " << gap_[i]->GetName() << std::endl;
  }
}

void RFAmplitudePVObserver::AttachBeamLineElement(BeamLineElement* r_elem)
{
  if (RFGap* tmp_gap = dynamic_cast<RFGap*>(r_elem))
    gap_.push_back(tmp_gap);
  else
  {
    std::cerr << "Cann't attach " << r_elem->GetName() 
      << " to a RFAmplitudePVObserver!" << std::endl;
    exit(-1);
  }
}

std::vector<std::string> RFAmplitudePVObserver::GetElementNames() const
{
  std::vector<std::string> rlt(gap_.size(), "");
  for(int i = 0; i < gap_.size(); ++i)
    rlt[i] = gap_[i]->GetName(); 
  return rlt; 
}
