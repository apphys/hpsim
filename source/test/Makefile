include ../make_root.mk
include ./make_local.mk

all:compile

list:
	@echo $(EXT_OBJS)

compile:$(TARGETS)
%:%.cpp 
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(PYTHON_LD_FLAGS) $(EPICS_LD_FLAGS) $(EXT_OBJS) -o $@ $^
clean:
	@-rm $(TARGETS)

.PHONY: compile clean
