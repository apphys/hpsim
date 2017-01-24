# config.py

eval_func = ''
num_objectives = 0
num_variables = 0
num_constraints = 0
population_size = 0
variable_min = []
variable_max = []

def init(num_obj, num_con, num_var, pop_sz, eval_f = ''):
    global num_objectives, num_variables, num_constraints, eval_func
    global population_size
    global variable_min, variable_max
    num_objectives = num_obj
    num_constraints = num_con
    num_variables = num_var
    population_size = pop_sz
    eval_func = eval_f
    variable_min = [float('-inf') for _ in range(num_variables)]
    variable_max = [float('inf') for _ in range(num_variables)]

def set_variable_limits(vmin, vmax, indx = -1):
    global variable_min, variable_max
    if indx >= 0 and indx < num_variables:
      variable_min[indx] = vmin
      variable_max[indx] = vmax
    elif indx < 0:
      for i in range(num_variables):
        variable_min[i] = vmin
        variable_max[i] = vmax
    else:
      raise IndexError("set_variable_limits index out of bound.")


