include ../make_root.mk
include ./make_local.mk

all:compile

compile:$(TARGETS)
%:%.cpp 
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(PYTHON_LD_FLAGS) $(EPICS_LD_FLAGS) $(GL_LD_FLAGS) $(EXT_OBJS) -o $@ $^
clean:
	@$(foreach t, $(TARGETS), if [ -f $(t) ]; then rm $(t); fi;)

.PHONY: compile clean
