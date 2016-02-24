include make_root.mk

DIR=$(filter-out doxygen/, $(filter-out test/, $(filter-out lib/, $(filter %/,$(shell ls -F)))))
OBJS=$(foreach d, $(filter-out server/, $(DIR)), $(wildcard $(d)obj/*.o))

all: compile ../bin/HPSim.so
list:
	@echo $(OBJS)
compile:
	@for d in $(DIR); \
	   do \
	     (cd $$d && (if [ -f ./Makefile ] ; then $(MAKE) compile ;fi)); \
	   done
	@if [ -d ../bin ]; then \
           rm -rf ../bin; fi; 
	@mkdir ../bin

../bin/HPSim.so:
	$(CC)  $(LDFLAGS) $(PYTHON_LD_FLAGS) -shared -o $@ $(OBJS)

clean:
	@for d in $(DIR); \
	   do \
	     (cd $$d && (if [ -d obj ] ; then rm -rf obj ;fi)); \
	   done
	@if [ -d ../bin ]; then \
           rm -rf ../bin; fi; 
	@(cd test && (if [ -f ./Makefile ] ; then $(MAKE) clean;fi));
.PHONY: compile clean
