include ../make_root.mk
include ./make_local.mk

vpath %.cpp $(LOC_SRC_DIR)
vpath %.cu $(LOC_SRC_DIR)

all:compile 

prep:
	@mkdir -p $(LOC_OBJ_DIR)
compile:prep $(LOC_OBJS) $(LOC_CUOBJS)
$(LOC_OBJ_DIR)/%.o:%.cpp
	$(CC) $(CPPFLAGS) $(LOCFLAGS) -c $< -o $@
$(LOC_OBJ_DIR)/%.cu.o:%.cu
	$(CC) $(CPPFLAGS) $(LOCFLAGS) -c $< -o $@
clean:
	@-rm $(LOC_OBJ_DIR)/*

.PHONY: compile clean
