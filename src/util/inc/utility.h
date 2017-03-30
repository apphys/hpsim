#ifndef UTILITY_H
#define UTILITY_H

#include <string>
#include <vector>

unsigned int NextPow2(unsigned int);

bool MatchCaseInsensitive(std::string r_input, std::string r_pattern);
double GetFirstNumberInString(std::string r_input);
bool StringEqualCaseInsensitive(std::string r_input1, std::string r_input2);
bool ContainNumbers(std::string r_input);
bool ContainOnlyNumbers(std::string r_str);
bool StartWithPattern(std::string r_line, std::string r_pattern);
void CleanString(std::string r_str);
std::vector<std::string> Split(std::string r_line);

#endif
