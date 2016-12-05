#ifndef SERVER_H
#define SERVER_H

#include "pv_observer_list.h"

extern bool server_stop_flag;
struct ServerArg
{
  int num_monitor;
  PVObserverList* channels;  
};

void* ServerRoutine(void*);
#endif
