#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cadef.h>
#include <epicsString.h>
#include <alarm.h>
#include "tool_lib.h"

static epicsTimeStamp tsStart;
static int tsInitC = 0;              /* Flag: Client timestamps init'd */
double caTimeout = 1.0;  /* wait time default (see -w option) */
capri caPriority = DEFAULT_CA_PRIORITY;  /* CA Priority */

int create_pvs (pv* pvs, int nPvs, caCh *pCB)
{
    int n;
    int result;
    int returncode = 0;

    if (!tsInitC)                /* Initialize start timestamp */
    {
        epicsTimeGetCurrent(&tsStart);
        tsInitC = 1;
    }
                                 /* Issue channel connections */
    for (n = 0; n < nPvs; n++) {
        result = ca_create_channel (pvs[n].name,
                                    pCB,
                                    &pvs[n],
                                    caPriority,
                                    &pvs[n].chid);
        if (result != ECA_NORMAL) {
            fprintf(stderr, "CA error %s occurred while trying "
                    "to create channel '%s'.\n", ca_message(result), pvs[n].name);
            pvs[n].status = result;
            returncode = 1;
        }
    }

    return returncode;
}
int connect_pvs (pv* pvs, int nPvs)
{
    int returncode = create_pvs ( pvs, nPvs, 0);
    if ( returncode == 0 ) {
                            /* Wait for channels to connect */
        int result = ca_pend_io (caTimeout);
        if (result == ECA_TIMEOUT)
        {
            if (nPvs > 1)
            {
                fprintf(stderr, "Channel connect timed out: some PV(s) not found.\n");
            } else {
                fprintf(stderr, "Channel connect timed out: '%s' not found.\n", 
                        pvs[0].name);
            }
            returncode = 1;
        }
    }
    return returncode;
}
