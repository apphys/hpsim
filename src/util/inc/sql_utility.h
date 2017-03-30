#ifndef SQL_UTILITY_H
#define SQL_UTILITY_H

#include <string>
#include <vector>
#include <utility>
#include <sqlite3.h>
#include "py_wrapper.h"

bool SQLCheck(int, std::string, char* r_err = NULL);
bool SQLCheck(int, sqlite3_stmt*, std::string, char* r_err = NULL);
void PrintSelectResult(sqlite3*, const char*);
std::vector<std::string> GetTableColumnNames(sqlite3* r_db, const char* r_sql);
std::string GetDataFromDB(sqlite3*, const char*);
std::vector<std::string> GetDataArrayFromDB(sqlite3*, const char*);
std::vector<std::pair<std::string, std::string> > GetDataPairArrayFromDB(
  sqlite3*, const char*);
std::vector<std::pair<std::string, std::pair<std::string, std::string> > > 
  GetDataTripletArrayFromDB(sqlite3*, const char*);
std::vector<std::vector<std::string> >
GetQueryResults(sqlite3* r_db, const char* r_sql);

/*!
 * \brief Database connection class.
 */
class DBConnection : public PyWrapper
{
public:
  DBConnection(std::string r_db_addr); 
  DBConnection(DBConnection& r_org);
  ~DBConnection();
  void LoadLib(std::string r_lib_addr);
  void AttachDB(std::string r_db_addr, std::string r_db_name);
  void PrintDBs() const;
  void PrintLibs() const;
  void ClearModelIndex();
  std::vector<std::string> GetEPICSChannels() const;
  //! Pointer to sqlite3 database connection
  sqlite3* db_conn;
  //! List of opened database aliases 
  std::vector<std::string> dbs;
  //! List of opened database with full addresses
  std::vector<std::string> db_addrs;
  //! List of external libraries
  std::vector<std::string> libs;
private:
  bool TableHasModelIndex(std::string r_tbl) const;
};


#endif
