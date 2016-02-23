#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include "utility.h"

unsigned int NextPow2(unsigned int r_x)
{
  --r_x;
  r_x |= r_x >> 1;
  r_x |= r_x >> 2;
  r_x |= r_x >> 4;
  r_x |= r_x >> 8;
  r_x |= r_x >> 16;
  return ++r_x;
}

bool MatchCaseInsensitive(std::string r_input, std::string r_pattern)
{
  std::string input = r_input;
  std::transform(input.begin(), input.end(), input.begin(), ::tolower);
  std::string pattern = r_pattern;
  std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);
  size_t found = input.find(pattern);
  if(found != std::string::npos)
    return true;
  else
    return false;
}

double GetFirstNumberInString(std::string r_input)
{
  size_t i = 0;
  bool num_start = false;
  bool has_dot = false;
  std::ostringstream osstr;
  while(i < r_input.size())
  {
    if(r_input[i] == '-' || r_input[i] == '+')
    {
      if(!num_start &&  i < r_input.size()-1 && ::isdigit(r_input[i+1]))
      {
        num_start = true;
        osstr << r_input[i];
      }
      else if(num_start)
        break;
    }
    else if(r_input[i] == '.')
    {
      if(num_start && !has_dot)
      {
        osstr << r_input[i];
        has_dot = true;
      }
      else if (has_dot)
        break;
    }
    else if(::isdigit(r_input[i]))
    {
      osstr << r_input[i];
      num_start = true;
    }
    else
    {
      if(num_start)
        break;
    }
    i++;
  }// while
  return std::atof(osstr.str().c_str());
}

bool StringEqualCaseInsensitive(std::string r_input1, std::string r_input2)
{
  std::string input1 = r_input1;
  std::transform(input1.begin(), input1.end(), input1.begin(), ::tolower);
  std::string input2 = r_input2;
  std::transform(input2.begin(), input2.end(), input2.begin(), ::tolower);

  if(input1 == input2)
    return true;
  else
    return false;
}

bool ContainNumbers(std::string r_input)
{
  size_t i = 0;
  while(i < r_input.size() && !::isdigit(r_input[i]))
    i++;
  if(i == r_input.size())
    return false;
  else
    return true;
}

bool ContainOnlyNumbers(std::string r_str)
{
  std::string str = r_str;
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  if(str.empty())
    return false;
  std::string::const_iterator it = str.begin();
  while(it != str.end()
        && (std::isdigit(*it)|| (*it)=='.' || (*it) == '-') || std::tolower(*it) == 'e')
    it++;
  return (it == str.end());
}

bool StartWithPattern(std::string r_line, std::string r_pattern)
{
  std::string str = r_line;
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  if(str.size() < r_pattern.size())
    return false;
  std::string tmp;
  tmp.assign(str, 0, r_pattern.size());
  if (StringEqualCaseInsensitive(tmp, r_pattern))
    return true;
  else
    return false;
}

void CleanString(std::string r_str)
{
  while(r_str.find(":") != std::string::npos)
    r_str.erase(r_str.find(":"), 1);
  while(r_str.find(",") != std::string::npos)
    r_str.erase(r_str.find(","), 1);
  while(r_str.find("\"") != std::string::npos)
    r_str.erase(r_str.find("\""), 1);
}

std::vector<std::string> Split(std::string r_line)
{
  std::vector<std::string> rlt;
  std::string tmp = r_line;
  CleanString(tmp);
  std::string delimiter = " ";
  size_t pos = 0;
  std::string token;
  uint cnt = 0;
  while((pos = tmp.find(delimiter)) != std::string::npos)
  {
    token = tmp.substr(0, pos);
    if(!token.empty() && cnt)
      rlt.push_back(token);
    tmp.erase(0, pos + delimiter.length());
    ++cnt;
  }
  if(!tmp.empty())
    rlt.push_back(tmp);
  return rlt;
}

