#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>

struct InputData
{
  std::string beam;
  std::vector<std::string> db;
  std::vector<std::string> libdb;
  std::string start;
  std::string end;
  bool server_off;
  std::vector<std::string> put_pvs;
};

InputData ProcessInputFile(std::string r_file);

#endif
