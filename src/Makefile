include make_root.mk

DIR=$(filter-out doxygen/, $(filter-out test/, $(filter-out server/, \
  $(filter %/,$(shell ls -F)))))
OBJS=$(foreach d, $(DIR), $(wildcard $(d)obj/*.o))
NONGRAPHICS_OBJS=$(foreach d, $(filter-out graphics/, $(DIR)), \
  $(wildcard $(d)obj/*.o))

all: compile ../bin/HPSim.so ../db/lib/libsqliteext.so
list:
	@echo $(OBJS)
compile:
	@if [ -d ../bin ]; then \
           rm -rf ../bin; fi; 
	@mkdir ../bin
	@for d in $(DIR); \
	   do \
	     (cd $$d && (if [ -f ./Makefile ] ; then $(MAKE) compile ;fi)); \
	   done

../bin/HPSim.so:
	$(CC)  $(LDFLAGS) $(PYTHON_LD_FLAGS) -shared \
		-o $@ $(NONGRAPHICS_OBJS)
#../bin/PyEPICS.so: server/obj/pyepics_module.o server/obj/ca_check.o
#	$(CC)  $(LDFLAGS) $(EPICS_LD_FLAGS) -shared -o $@ $^
../db/lib/libsqliteext.so: 
	gcc -shared -o ../db/lib/libsqliteext.so ../db/lib/extension-functions.c -lm

clean:
	@for d in $(DIR); \
	   do \
	     (cd $$d && (if [ -d obj ] ; then rm -rf obj ;fi)); \
	   done
	@if [ -d ../bin ]; then \
           rm -rf ../bin; fi; 
	@(cd test && (if [ -f ./Makefile ] ; then $(MAKE) clean;fi));
.PHONY: compile clean
