#include <errno.h> //for EBUSY 
#include <omp.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cadef.h>
#include "server.h"
#include "ca_check.h"
#include "pthread_check.h"
#include "tool_lib.h"

bool server_stop_flag = false;
PVList* pv_channels;

void event_handler(evargs args)
{
  pv* ppv = static_cast<pv*>(args.usr);

  ppv->status = args.status;
  if (args.status == ECA_NORMAL)
  {
    ppv->dbrType = args.type;
    memcpy(ppv->value, args.dbr, dbr_size_n(args.type, args.count));
    double val = ((double*)dbr_value_ptr(ppv->value, ppv->dbrType))[0];
    printf("epics update: %s = %f \n", ppv->name, val);
    fflush(stdout);
    std::ostringstream osstr;
    osstr << val;

    Updater* updater = (*pv_channels)[std::string(ppv->name)];
//    ThreadCheck(pthread_mutex_lock(&updater_pair.second), "EPICSMonitorRoutine:lock mutex");
    updater->SetValue(osstr.str());
    updater->UpdateDB();
    updater->UpdateModel();
    std::cout << ppv->name << " after setting val old : " << 
      updater->old_val_ << ", new = " << 
      updater->val_ << std::endl;
    updater->UpdateOldValue();
//    ThreadCheck(pthread_mutex_unlock(&updater_pair.second), "EPICSMonitorRoutine:unlock mutex");
  }
}

void connection_handler(struct connection_handler_args args)
{
  pv *ppv = ( pv * ) ca_puser ( args.chid );
  if ( args.op == CA_OP_CONN_UP ) {
                                /* Set up pv structure */
                                /* ------------------- */

                                /* Get natural type and array count */
        ppv->nElems  = ca_element_count(ppv->chid);
        ppv->dbfType = ca_field_type(ppv->chid);
        ppv->dbrType = DBR_TIME_DOUBLE;
        ppv->onceConnected = 1;
       if ( ! ppv->value ) {
                                    /* Allocate value structure */
            ppv->value = calloc(1, dbr_size_n(ppv->dbrType, ppv->reqElems));
            if ( ppv->value ) {
                evid evid;
                ppv->status = ca_create_subscription(ppv->dbrType,
                                                ppv->reqElems,
                                                ppv->chid,
                                                DBE_VALUE,
                                                event_handler,
                                                (void*)ppv,
                                                &evid);

                if ( ppv->status != ECA_NORMAL ) {
                    printf("create EPICS subscription failed.");
                    free ( ppv->value );
                }
            }
        }
  }
  else if ( args.op == CA_OP_CONN_DOWN ) {
        ppv->status = ECA_DISCONN;
        printf("pv %s disconnected. \n", ppv->name);

  }
}

bool InitChannel(Updater* r_up)
{
  std::string pv_name = r_up->GetPV();
  chid mychid;
  double val_dbl = 0.0;
  if(!CACheck(ca_create_channel(pv_name.c_str(),NULL,NULL,10,&mychid), "ca_create_channel", pv_name))
    return false;
  if(!CACheck(ca_pend_io(10.0), "ca_pend_io after ca_create_channel", pv_name))
    return false;
  if(!CACheck(ca_get(DBR_DOUBLE, mychid, (void*)&val_dbl), "ca_get", pv_name))
    return false;
  if(!CACheck(ca_pend_io(10.0), "ca_pend_io after ca_get", pv_name))
    return false;
  if(!CACheck(ca_clear_channel(mychid), "ca_clear_channel", pv_name))
    return false;

  std::ostringstream osstr;
  osstr << val_dbl;
  r_up->SetValue(osstr.str());
  return true;
}

void* ServerRoutine(void* r_arg)
{
  // EPICS ca context setup
  CACheck(ca_context_create(ca_disable_preemptive_callback),"ServerRoutine:ca_context_create");

  ServerArg* arg = (ServerArg*)r_arg;
  pv_channels = arg->channels; 
  std::cout <<"server is monitoring " << pv_channels->GetListSize() << " pvs" << std::endl;
  
  std::map<std::string, Updater*>::iterator iter;

  const int num_channel = pv_channels->GetListSize();
  pv* pvs = new pv[num_channel];
  int cnt = 0;
  for (iter = (pv_channels->list).begin(); iter != (pv_channels->list).end(); ++iter)
  {
    pvs[cnt++].name = const_cast<char*>((iter->first).c_str());
    InitChannel((iter->second));
  }
    

  create_pvs(pvs, num_channel, connection_handler);
  ca_pend_event(DEFAULT_TIMEOUT);
  for(int i = 0; i < num_channel; ++i)
  {
    if(!pvs[i].onceConnected)
      std::cerr << "Cannot connect to " << pvs[i].name << std::endl;
    else 
      std::cout << pvs[i].name << " connected." << std::endl;
  }

  while(!server_stop_flag)
  {
    ca_pend_event(1e-9);
    usleep(10000);
  }
  std::cout << "closing EPICS monitor..." << std::endl;

  void* status;
  ca_context_destroy();
  std::cout << "Data server closed." << std::endl;
  pthread_exit(NULL);
}
