#ifndef PV_OBSERVER_H
#define PV_OBSERVER_H

#include <string>
#include <vector>
#include <sqlite3.h>
#include "beamline.h"

class PVObserver
{
public:
  PVObserver(std::string, std::string); // name, db
  std::string GetPV() const;
  std::string GetDB() const;
  sqlite3* GetDBconn() const;
  void SetPV(std::string r_pv);
  void SetDB(std::string r_db);
  void SetDBconn(sqlite3* r_db_conn);
  void Update(std::string);
  virtual void AttachBeamLineElement(BeamLineElement*) = 0;
  virtual std::vector<std::string> GetElementNames() const = 0;
private:
  void UpdateDB();
  virtual void UpdateModel() = 0;
  std::string pv_;
  std::string val_;
  std::string db_;
  sqlite3* db_conn_;
};

class QuadPVObserver : public PVObserver
{
public:
  QuadPVObserver(std::string, std::string);
  void AttachBeamLineElement(BeamLineElement*);
  std::vector<std::string> GetElementNames() const;
private:
  void UpdateModel();
  std::vector<Quad*> quad_;
};

class RFPhasePVObserver : public PVObserver
{
public:
  RFPhasePVObserver(std::string, std::string);
  void AttachBeamLineElement(BeamLineElement*);
  std::vector<std::string> GetElementNames() const;
private:
  void UpdateModel();
  std::vector<RFGap*> gap_;
};

class RFAmplitudePVObserver : public PVObserver
{
public:
  RFAmplitudePVObserver(std::string, std::string);
  void AttachBeamLineElement(BeamLineElement*);
  std::vector<std::string> GetElementNames() const;
private:
  void UpdateModel();
  std::vector<RFGap*> gap_;
};

inline
std::string PVObserver::GetPV() const
{
  return pv_;
}
inline
std::string PVObserver::GetDB() const
{
  return db_;
}
inline
sqlite3* PVObserver::GetDBconn() const
{
  return db_conn_;
}
inline
void PVObserver::SetPV(std::string r_pv)
{
  pv_ = r_pv;
}
inline
void PVObserver::SetDBconn(sqlite3* r_db_conn)
{
  db_conn_ = r_db_conn;
}
inline
void PVObserver::SetDB(std::string r_db)
{
  db_ = r_db;
}
#endif
