# lcsutil_v2.py
# Collection of python functions for accessing LANSCE Control System (LCS) info
# 20160223 - ljr - code cleanup

def expand_pv(pv):
    """ Returns the LCS Process Variable in full syntax AADDDNNNCMM as a string
    where AA is two character area,
          DD is device type,
          NNN is device number,
          C  is channel type,
          MM is channel number 
    Returns partial PVs when input string is incomplete.

    Argument:
       pv(str): LCS PV to expane

    """
    # Parse pv
    index = 0
    # Get AA
    if pv[index].isdigit():
        if pv[index+1].isdigit():
            aa = pv[index:2].zfill(2)
            index = 2
        else:
            aa = pv[0].zfill(2)
            index = 1
    else:
        aa = pv[0:2].upper()
        index = 2
    # Get DD
    dd = pv[index:index+2].upper()
    index += 2
    # Get NNN
    ndx = 0
    if index < len(pv):
        while index + ndx < len(pv) and pv[index + ndx].isdigit():
            ndx += 1
        nnn = pv[index: index + ndx].zfill(3)
    else:
        nnn = ""
    # Get C
    index += ndx
    if index < len(pv):
        c = pv[index:index + 1].upper()
        index += 1
    else:
        c = ""
    # Get MM
    if index < len(pv):
        mm = pv[index:].zfill(2) 
    else:
        mm = ""
    return aa + dd + nnn + c + mm

def get_pv_area(pv):
    """Returns two character area for PV

    Arguments:
       pv(str): name of LCS PV
    """
    return expand_pv(pv)[0:2]
  
def get_pv_device(pv):
    """Returns two character device for PV

    Arguments:
       pv(str): name of LCS PV
    """
    return expand_pv(pv)[2:4]
  
def get_pv_type(pv):
    """Returns one character type for PV

    Arguments:
       pv(str): name of LCS PV
    """
    return expand_pv(pv)[7:8]
  
def get_pv_psp(n, beam='+'):
    """Returns EPICS/LCS PV name (string) for the phase set point of 
    module n or buncher n can be a string e.g. 'TA' or '05', or a number 5.

    Arguments:
       n(str or int): module or buncher number
       beam(str, optional): beam species
    """
    # Make two character area from input
    ns=str(n).upper().zfill(2)
    if ns == 'TA' or (ns == 'PB' and beam in get_pos_beams()):
        psp = expand_pv('tadb1e2')
    elif ns == 'TB' or (ns == 'PB' and beam in get_neg_beams()):
        psp = expand_pv('tbdb2e2')
    elif ns == 'TD' or ns == 'MB':
        psp = expand_pv('tddb1e2')
    elif ns in ['T1', 'T2', 'T3', 'T4']:
        psp = expand_pv(ns[1].zfill(2)+'js1d2')
    elif int(ns) < 5:
        psp = expand_pv(ns+'js1d2')
    elif int(ns) > 4:
        psp = expand_pv(ns+'ks1e1')
    return psp

def get_pos_beams():
    """ Returns list of names associated with H+ beam:
    ['TA', 'H+', 'LA', 'IP', '+']"""
    return ['TA', 'H+', 'LA', 'IP', '+']

def get_neg_beams():
    """ Returns list of names associated with H- beam
    ['TB', 'H-', 'LB', '-']"""
    return ['TB', 'H-', 'LB', '-']

def get_pv_asp(n, beam='+'):
    """Returns EPICS/LCS PV name (string) for the amplitude set point of 
    module n or buncher n,beam '+' or '-', n can be a string 
    e.g. 'TA' or '05', or a number 5.

    Arguments:
       n(str or int): module or buncher number
       beam(str, optional): beam species
    """
    # Make two character area
    ns = str(n).upper().zfill(2) 
    if ns == 'TA' or (ns == 'PB' and beam in get_pos_beams()):
        asp = expand_pv('tadb1e1')
    elif ns == 'TB'or (ns == 'PB' and beam in get_neg_beams()):
        asp = expand_pv('tbdb2e4')
    elif ns == 'TD' or ns == 'MB':
        if beam in get_pos_beams():
            asp = expand_pv('tddb1e7')
        elif beam in get_neg_beams():
            asp = expand_pv('tddb1e4')
        else:
            "Beam must be either", pos, " or ", neg
    elif ns in ['T1', 'T2', 'T3', 'T4']:
        asp = expand_pv(ns[1].zfill(2)+'js1d1')
    elif int(ns) < 5:
        asp = expand_pv(ns+'js1d1')
    elif int(ns) > 4:
        asp = expand_pv(ns+'ks1e2')
    return asp

def get_pv_asp_n(n):
    """Returns ASP PV associated with neg(-) beam species, either -.
    Required for Main Buncher.

    Arguments:
       n(str or int): module or buncher number
    """
    return get_pv_asp(n, '+')

def get_pv_asp_p(n):
    """Returns ASP PV associated with pos(+) beam species, either +.
    Required for Main Buncher.

    Arguments:
       n(str or int): module or buncher number
    """
    return get_pv_asp(n, '-')

def get_pv_rf(n, beam='+'):
    """Returns Epics/LCS PV name (string) for the on(intime)/off(delayed) 
    set point of module n or buncher n, where n can be a string 
    e.g. 'TA' or '05', or a number 5.

    Arguments:
       n(str or int): module or buncher number
       beam(str, optional): beam species
    """
    # Make two character area from input
    ns = str(n).upper().zfill(2) 
    if ns == 'TA' or (ns == 'PB' and beam in get_pos_beams()):
#    if ns == 'TA': #or ns == 'PB':
        rf = expand_pv('tadb1l3')
    elif ns == 'TB'or (ns == 'PB' and beam in get_neg_beams()):
#    elif ns == 'TB': #or ns == 'PB':
        rf = expand_pv('tbdb2l3')
    elif ns == 'TD'or ns == 'MB':
        rf = expand_pv('tddb1l3')
    elif ns in ['T1', 'T2', 'T3', 'T4']:
        rf = expand_pv(ns[1].zfill(2)+'tm1l1')
    elif int(ns) < 5:
        rf = expand_pv(ns+'tm1l1')
    elif int(ns) > 4:
        rf = expand_pv(ns+'tm1l1')
    return rf

def rf_on_val(n):
    """Returns RF 'ON' value for bunchers, DTL or CCL rf,
    n can be a string e.g. 'TA' or '05', or a number 5.

    Arguments:
       n(str or int): module or buncher number
    """
    # Make two character area from input
    area = str(n).upper().zfill(2) 
    if area in ['PB', 'MB', 'TA', 'TB', 'TD']:
        ## ON => RF Drive is ON
        ## previously:  rf_val = 'ON', but now all numeric
        # 0 => RF Drive is ON
        rf_val = 0
    else:
        ## ON => RF Delay is NO
        ## previously:  rf_val = 'ON', but now all numeric
        # 0 => RF is IN-TIME
        rf_val = 0
    return rf_val

def rf_off_val(n):
    """Returns RF 'OFF' value for bunchers, DTL or CCL rf,
    n can be a string e.g. 'TA' or '05', or a number 5.

    Arguments:
       n(str or int): module or buncher number
    """
    # Make two character area
    area = str(n).upper().zfill(2) 
    if area in ['PB', 'MB', 'TA', 'TB', 'TD']:
        ## OFF => RF Drive is OFF
        ## previously:  rf_val = 'OFF', but now all numeric
        # 1 => RF Drive is OFF
        rf_val = 1
    else:
        ## OFF => RF Delay is YES
        ## previously:  rf_val = 'YES', but now all numeric
        # 1 => RF is DELAYED
        rf_val = 1
    return rf_val
