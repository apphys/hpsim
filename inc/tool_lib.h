#ifndef MY_EPICS_TOOL_LIB_H
#define MY_EPICS_TOOL_LIB_H

#include <epicsTime.h>

#define DEFAULT_CA_PRIORITY 0  /* Default CA priority */
#define DEFAULT_TIMEOUT 1.0     /* Default CA timeout */

typedef chid                    chanId; /* for when the structures field name is "chid" */

typedef struct
{
    char* name;
    chanId chid;
    long  dbfType;
    long  dbrType;
    unsigned long nElems;
    unsigned long reqElems;
    int status;
    void* value;
    epicsTimeStamp tsPreviousC;
    epicsTimeStamp tsPreviousS;
    char firstStampPrinted;
    char onceConnected;
} pv;

extern double caTimeout;    /* Wait time default (see -w option) */
extern int  create_pvs (pv *pvs, int nPvs, caCh *pCB );
extern int  connect_pvs (pv *pvs, int nPvs );

#endif
