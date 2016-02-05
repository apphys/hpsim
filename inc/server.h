#ifndef SERVER_H
#define SERVER_H

#include "pvlist.h"

extern bool server_stop_flag;
struct ServerArg
{
  int num_monitor;
  PVList* channels;  
};

void* ServerRoutine(void*);
#endif
