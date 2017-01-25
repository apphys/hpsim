#!/usr/bin/env python

'''
run MOPSO
'''
import random
import math
import mo

def my_eval(ind):
#      print '--- evaluate ---'
      var1 = ind.variables[0]
      var2 = ind.variables[1]
      ind.objectives[0] = var1*var1 - var2
      ind.objectives[1] = -(0.5*var1 + var2 + 1.0)
      ind.constraints[0] = 6.5 - var1/6.0 - var2
      ind.constraints[1] = 7.5 - var1/2.0 - var2
      ind.constraints[2] = 30 - var1*5.0 - var2

      if ind.num_constraints == 0:
        ind.constraint_violation = 0.0
      else:
        ind.constraint_violation = 0.0
        for c in ind.constraints:
          if c < 0.0:
            ind.constraint_violation += c 
   
if __name__ == '__main__':
  mo.init(num_obj = 2, num_con=3, num_var=2, pop_sz = 50, eval_f = my_eval)
  mo.set_variable_limits(vmin = 0, vmax = 7)
  mp = mo.mopso(archive_size = 50, mutation_rate = 0.1)
  mp.run(num_generation = 10)
