#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sqlite3.h>
#include <vector>
#include <iterator>
#include "sql_utility.h" 
#include "utility.h" 

bool SQLCheck(int r_status, std::string r_op, char* r_err) 
{
  if(r_status != SQLITE_OK)
  {
    std::cerr << "SQL error: " << r_op << std::endl;
    if(r_err != NULL)
    {
      std::cerr << "Error Msg: " << std::string(r_err) << std::endl;
      sqlite3_free(r_err);
    }
    else 
      std::cerr << std::endl;
    return false;
  }
  return true;
}

bool SQLCheck(int r_status, sqlite3_stmt* r_stmt, std::string r_op, char* r_err) 
{
  if(r_status != SQLITE_OK)
  {
    std::cerr << "SQL error: " << r_op << " failure" ;  
    sqlite3* db = sqlite3_db_handle(r_stmt);
    std::cerr <<", Error: " << sqlite3_errmsg(db) << std::endl; 
    if(r_err != NULL)
      sqlite3_free(r_err);
    return false;
  }
  return true;
}

void PrintSelectResult(sqlite3* r_db, const char* r_sql)
{
  std::cout << "PrintSelectResult: " << r_sql << std::endl; 
  char **result, *err = NULL;
  int rc, i, j, k, l, nrows, ncols, width, *widths; 

  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "PrintSelectResult():\n" + std::string(r_sql), err);
  /* Determine column widths*/

  widths = (int*) malloc(ncols * sizeof(int));
  memset(widths, 0, ncols * sizeof(int));
  for(i = 0; i <= nrows; i++) {
      for(j = 0; j < ncols; j++) {
          if(result[i * ncols + j] == NULL)
              continue;
          width = strlen(result[i * ncols + j]);
          if(width > widths[j]) {
              widths[j] = width;
          }
      }
  }

  for(i = 0; i <= nrows; i++) {
      if(i == 1) {
          for(k = 0; k < ncols; k++) {
              for(l = 0; l < widths[k]; l++) 
                  std::cout << "-";
              std::cout << " ";
          }
          std::cout << std::endl;
      }

      for(j = 0; j < ncols; j++) 
          fprintf(stdout, "%-*s", widths[j] + 1, result[i * ncols + j]);
      std::cout << std::endl;
  }
  free(widths);

  sqlite3_free_table(result);
}

std::string GetDataFromDB(sqlite3* r_db, const char* r_sql)
{
  std::string rt = "";
  char **result, *err = NULL;
  int nrows = 0, ncols = 0;
  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "GetDataFromDB():\n" + std::string(r_sql), err); 
  if(nrows > 0 && result[1] != NULL)
    rt = std::string(result[1]);
  sqlite3_free_table(result);
  return rt;
}

std::vector<std::string>
GetDataArrayFromDB(sqlite3* r_db, const char* r_sql)
{
  std::vector<std::string> rt;
  char **result, *err = NULL;
  int nrows, ncols;
  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "GetDataArrayFromDB():\n" + std::string(r_sql), err); 
  for(int i = 1; i <= nrows; ++i)
  {
    if(result[i] != NULL) 
      rt.push_back(std::string(result[i]));
  }
  sqlite3_free_table(result);
  return rt;
}

std::vector<std::string>
GetTableColumnNames(sqlite3* r_db, const char* r_sql)
{
  std::vector<std::string> rt;
  char **result, *err = NULL;
  int nrows, ncols;
  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "GetTableColumnNames():\n" + std::string(r_sql), err); 
  for(int j = 0; j < ncols; j++) 
    rt.push_back(std::string(result[j]));
  sqlite3_free_table(result);
  return rt;
}

std::vector<std::vector<std::string> >
GetQueryResults(sqlite3* r_db, const char* r_sql)
{
  std::vector<std::vector<std::string> > rt;
  char **result, *err = NULL;
  int nrows = 0, ncols = 0;
  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "GetQueryResults():\n" + std::string(r_sql), err); 
  std::vector<std::string> arow;
  for(int i = 1; i <= nrows; ++i)
  {
    arow.resize(0);
    for(int j = 0; j < ncols; ++j)
      arow.push_back(result[ncols * i + j] != NULL ? 
                    std::string(result[ncols * i + j]) : "");
    rt.push_back(arow); 
  }
  sqlite3_free_table(result);
  return rt;
}

std::vector<std::pair<std::string, std::string> > 
GetDataPairArrayFromDB(sqlite3* r_db, const char* r_sql)
{
  std::vector<std::pair<std::string, std::string> > rt;
  char **result, *err = NULL;
  int nrows, ncols;
  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "GetDataPairArrayFromDB():\n" + std::string(r_sql), err); 
  std::string first_str, second_str;
  for(int i = 1; i <= nrows; ++i)
  {
    if(result[2 * i] != NULL) 
      first_str = std::string(result[2*i]);
    else 
      first_str = ""; 
    if(result[2 * i + 1] != NULL)
      second_str = std::string(result[2 * i + 1]);
    else
      second_str = "";
    rt.push_back(std::make_pair(first_str, second_str));
  }
  sqlite3_free_table(result);
  return rt;
}

std::vector<std::pair<std::string, std::pair<std::string, std::string> > > 
  GetDataTripletArrayFromDB(sqlite3* r_db, const char* r_sql)
{
  std::vector<std::pair<std::string, std::pair<std::string, std::string> > > rt;
  char **result, *err = NULL;
  int nrows, ncols;
  SQLCheck(sqlite3_get_table(r_db, r_sql, &result, &nrows, &ncols, &err), 
    "GetDataTripletArrayFromDB():\n" + std::string(r_sql), err); 
  std::string first_str, second_str, third_str;
  for(int i = 1; i <= nrows; ++i)
  {
    if(result[3 * i] != NULL) 
      first_str = std::string(result[3 * i]);
    else 
      first_str = ""; 
    if(result[3 * i + 1] != NULL)
      second_str = std::string(result[3 * i + 1]);
    else
      second_str = "";
    if(result[3 * i + 2] != NULL)
      third_str = std::string(result[3 * i + 2]);
    else
      third_str = "";
    rt.push_back(std::make_pair(first_str, 
      std::make_pair(second_str, third_str)));
  }
  sqlite3_free_table(result);
  return rt;
}

/*!
 * \brief Constructor. Open the sqlite3 connection & enable loading external 
 * 	libraries.
 * \param r_db_addr Name of the database with full address
 */
DBConnection::DBConnection(std::string r_db_addr) : PyWrapper()
{
  sqlite3_open(r_db_addr.c_str(), &db_conn);
  SQLCheck(sqlite3_enable_load_extension(db_conn, 1), 
    "DBConnection::LoadLib : sqlite3_enable_extension");   
  dbs.push_back("main");
  db_addrs.push_back(r_db_addr);
}

/*!
 * \brief Destructor. Closing the sqlite3 connection.
 */
DBConnection::~DBConnection()
{
  sqlite3_close(db_conn);
}

/*!
 * \brief Copy constructor.
 */
DBConnection::DBConnection(DBConnection& r_org) : PyWrapper()
{
  if(!r_org.dbs.empty())
  {
    sqlite3_open(r_org.db_addrs[0].c_str(), &db_conn);
    SQLCheck(sqlite3_enable_load_extension(db_conn, 1), 
      "DBConnection::LoadLib : sqlite3_enable_extension");   
    db_addrs.push_back(r_org.db_addrs[0]);
    dbs.push_back(r_org.dbs[0]);
    for(int i = 1; i < r_org.dbs.size(); ++i)
      AttachDB(r_org.db_addrs[i], r_org.dbs[i]);
    if(!r_org.libs.empty())
      for(int i = 0; i < r_org.libs.size(); ++i)
        LoadLib(r_org.libs[i]);
  }
}

/*!
 * \brief Load the external library. 
 */
void DBConnection::LoadLib(std::string r_lib_addr)
{
  char* errmsg;
  if(!SQLCheck(sqlite3_load_extension(db_conn, r_lib_addr.c_str(), 0, &errmsg), 
    "DBConnection::LoadLib : sqlite3_load_extension"))
    std::cout << "sqlite3_load_extension error msg :" << errmsg << std::endl;
  libs.push_back(r_lib_addr);
}

/*!
 * \brief Attach a database to the connection.
 * \param r_db_addr Name of the database to be attached with full address
 * \param r_db_name Alias of the database
 */
void DBConnection::AttachDB(std::string r_db_addr, std::string r_db_name)
{
  std::string sql = "attach '" + r_db_addr + "' as " + r_db_name;
  char* errmsg;
  SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), 
    "DBConnection::AttachDB : sqlite3_exec");
  dbs.push_back(r_db_name);
  db_addrs.push_back(r_db_addr);
}

void DBConnection::PrintDBs() const
{
  if(dbs.empty())
  {
    std::cout << "DBConnection is not connected to any db."<< std::endl;
    return;
  }
  for(int i = 0; i < dbs.size(); ++i)
  {
    std::cout << dbs[i] << " -> ";
    std::cout << db_addrs[i] << std::endl;
  } 
}

void DBConnection::PrintLibs() const
{
  if(libs.empty())
  {
    std::cout << "No lib is loaded for this DBConnection."<< std::endl;
    return;
  }
  for(int i = 0; i < libs.size(); ++i)
    std::cout << libs[i] << std::endl;
}

/*!
 * \brief Check if a table in sqlite3 has a field called "model_index"
 * \param r_tbl Name of the sqlite3 table. 
 */
bool DBConnection::TableHasModelIndex(std::string r_tbl) const
{
  std::string sql = "pragma table_info(" + r_tbl + ")";
  char **result, *err;
  int nrows, ncols;
  SQLCheck(sqlite3_get_table(db_conn, sql.c_str(), &result, &nrows, &ncols, 
            &err), "DBConnection::TableHasModelIndex()\n" + sql, err);
  for(int i=0; i <= nrows; i++) 
  {
    for(int j=0; j < ncols; j++) 
    {
      if(result[i * ncols + j] == NULL) continue;
      std::string tmp = result[i * ncols + j];
      size_t found = tmp.find("model_index");
      if(found != std::string::npos)
      {
        sqlite3_free_table(result);
        return true;
      }
    }
  }
  sqlite3_free_table(result);
  return false;
}

/*!
 * \brief Clear the model_index field for all the tables containing
 * 	this field.
 */
void DBConnection::ClearModelIndex()
{
  char* errmsg;
  for(int i = 0; i < dbs.size(); ++i)
  {
    std::string sql = "select name from " + dbs[i] + 
      ".sqlite_master where type = 'table'";
    std::vector<std::string> tbls = GetDataArrayFromDB(db_conn, sql.c_str());
    for(int j = 0; j < tbls.size(); ++j)
    {
      if(TableHasModelIndex(tbls[j]))
      {
        sql = "update " + dbs[i] + "." + tbls[j] + " set model_index = NULL";
        SQLCheck(sqlite3_exec(db_conn, sql.c_str(), NULL, NULL, &errmsg), 
			      "DBConnection::ClearModelIndex : " 
                              + dbs[i] + "." + tbls[j]);
      }
    }
  }
}

/*!
 * \brief Get a list of EPICS PVs in all the database attached.
 */
std::vector<std::string> DBConnection::GetEPICSChannels() const
{
  std::vector<std::string> rt;
  for(int dbs_indx = 0; dbs_indx < dbs.size(); ++dbs_indx)
  {
    std::string sql = "select lcs_name from " + dbs[dbs_indx] + 
      ".epics_channel"; 
    std::vector<std::string> pv_list = GetDataArrayFromDB(db_conn, sql.c_str());
    std::copy(pv_list.begin(), pv_list.end(), std::back_inserter(rt));
  }
  return rt;
}

/*
int GetQueryCallback(void *arg, int count, char **data, char **columns)
{
  std::vector<std::vector<std::string> >* tb = 
    static_cast<std::vector<std::vector<std::string> >*>(arg);
  std::vector<std::string> arow;
  for(int i = 0; i < count; ++i)
    arow.push_back(std::string(data[i]));
  tb->push_back(arow);
  return 0;
}

std::vector<std::vector<std::string> >
GetQueryResults2(sqlite3* r_db, const char* r_sql)
{
  std::vector<std::vector<std::string> > rt;
  std::cout << "GetQueryResults2: " << r_sql << std::endl;
  SQLCheck(sqlite3_exec(r_db, r_sql, GetQueryCallback, &rt, NULL), 
    "GetQueryResults2():\n"+ std::string(r_sql));
  return rt;
}
*/
