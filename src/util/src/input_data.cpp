#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include "input_data.h"
#include "utility.h"

/*!
 * \brief Process the input txt file for online mode.
 */
InputData ProcessInputFile(std::string r_file)
{
  InputData rlt; 
  std::ifstream input(r_file.c_str());
  std::string tmp;
  bool db_not_found = true;
  bool beam_not_found = true;
  bool start_not_found = true;
  bool end_not_found = true;
  while(std::getline(input, tmp))
  {
    if(StartWithPattern(tmp, "db"))
    {
      rlt.db = Split(tmp);
      if(!rlt.db.empty()) 
      {
        std::cout << "DB files: ";
        std::copy(rlt.db.begin(), rlt.db.end(), 
	  std::ostream_iterator<std::string>(std::cout, "  "));
        std::cout << std::endl;
        db_not_found = false;
      }
      else
      {
        std::cerr << "No db files! "<< std::endl;
        exit(-1);
      }
    }
    if(StartWithPattern(tmp, "libdb"))
    {
      rlt.libdb = Split(tmp);       
      std::cout << "DB external libraries: " ;
      std::copy(rlt.libdb.begin(), rlt.libdb.end(), 
	std::ostream_iterator<std::string>(std::cout, "  "));
      std::cout << std::endl;
    }
    if(StartWithPattern(tmp, "beam"))
    {
      std::vector<std::string> beam_tmp = Split(tmp); 
      if(!beam_tmp.empty()) 
      {
        rlt.beam = beam_tmp[0];
        beam_not_found = false;
        std::cout << "Beam file: " << rlt.beam << std::endl;
      }
      else
      {
        std::cerr << "No input beam file! "<< std::endl;
        exit(-1);
      }
    }
    if(StartWithPattern(tmp, "start"))
    {
      std::vector<std::string> start_tmp = Split(tmp); 
      if(!start_tmp.empty()) 
      {
        rlt.start = start_tmp[0];
        start_not_found = false;
        std::cout << "start: " << rlt.start << std::endl;
      }
      else
      {
        rlt.start = "TADB01";
        std::cout << "No start is defined, use default: TADB01 "<< std::endl;
      }
    }
    if(StartWithPattern(tmp, "end"))
    {
      std::vector<std::string> end_tmp = Split(tmp); 
      if(!end_tmp.empty()) 
      {
        rlt.end = end_tmp[0];
        end_not_found = false;
        std::cout << "end: " << rlt.end<< std::endl;
      }
      else
      {
        rlt.end = "TREM01";
        std::cout << "No end is defined, use default: TREM01 "<< std::endl;
      }
    }
    if(StartWithPattern(tmp, "server"))
    {
      rlt.server_off = true;
      std::vector<std::string> server_tmp = Split(tmp);
      if(!server_tmp.empty())
      {
        if(StringEqualCaseInsensitive(server_tmp[0], "on")) 
          rlt.server_off = false;
        std::cout << "server: " << server_tmp[0] << std::endl;
      }
      else
        std::cout << "No server status is defined, use default: off " 
	  << std::endl;
    }
    if(StartWithPattern(tmp, "output"))
      rlt.put_pvs =  Split(tmp);
  }

  if(db_not_found)
  {
    std::cerr << "No db files! " << std::endl; exit(-1);
  }
  if(beam_not_found)
  {
    std::cerr << "No input beam file! " << std::endl; exit(-1);
  }
  if(start_not_found)
  {
    std::cout << "No start is defined, use default: TADB1" << std::endl;
    rlt.start = "TADB1";
  }
  if(end_not_found)
  {
    std::cout << "No end is defined, use default: TREM1" << std::endl;
    rlt.end= "TREM1";
  }
  return rlt;
}
