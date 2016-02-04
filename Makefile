CXX=nvcc
CC=nvcc
SRCDIR=src
OBJDIR=obj
EPICS_INC_FLAGS=-I/ade/epics/supTop/base/R3.14.11/include/os/Linux -I/ade/epics/supTop/base/R3.14.11/include -D_POSIX_C_SOURCE=199506L -D_POSIX_THREADS -D_XOPEN_SOURCE=500 -D_X86_64_  -DUNIX  -D_BSD_SOURCE -Dlinux  -D_REENTRANT -m64 
PYTHON_INC_FLAGS=-I/usr/include/python2.6
#CPPFLAGS+=$(EPICS_INC_FLAGS)
CPPFLAGS+=$(PYTHON_INC_FLAGS)
CPPFLAGS+=-I./inc -arch=sm_35 -Xcompiler '-fPIC' -O3 -g -w #-DDOUBLE_PRECISION #-D_DEBUG
#CPPFLAGS+=-Xptxas -v -Xptxas -dlcm=ca -g
EPICS_LD_FLAGS=-L/ade/epics/supTop/base/R3.14.11/lib/linux-x86_64 -lcas -lgdd -lasHost -ldbStaticHost -lregistryIoc -lca -lCom -lpthread -lreadline -lncurses -lm -lrt -ldl -lgcc
GL_LD_FLAGS+=-L./lib -lglut -lGLEW_x86_64 -lGLU
PYTHON_LD_FLAGS=-L./usr/lib/python2.6/config/ -lpython2.6 -lpthread
LDFLAGS=-lcurand -Xcompiler '-fopenmp' -lsqlite3

SRCS:=$(notdir $(wildcard $(SRCDIR)/*.cpp))
OBJS=$(addprefix ./obj/,$(SRCS:.cpp=.o))
CUSRCS:=$(notdir $(wildcard $(SRCDIR)/*.cu))
CUOBJS=$(addprefix ./obj/,$(CUSRCS:.cu=.cu.o))
GRAPHICS_OBJS=./obj/graphics_2d.o ./obj/graphics_3d.o ./obj/graphics_common_func.o
NONGRAPHICS_OBJS:=$(filter-out $(GRAPHICS_OBJS), $(OBJS))
GRAPHICS_CUOBJS=./obj/graphics_kernel_call.cu.o
NONGRAPHICS_CUOBJS:=$(filter-out $(GRAPHICS_CUOBJS),$(CUOBJS))

vpath %.h ./inc
vpath %.cpp ./src
vpath %.cu ./src


#all: ./packages/HPSim.so ./packages/PyEPICS.so lib/libsqliteext.so start2d startalex
all: run

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
$(OBJDIR)/%.cu.o: %.cu
	$(CXX) $(CPPFLAGS) -c $< -o $@
run: $(NONGRAPHICS_OBJS) $(NONGRAPHICS_CUOBJS) main.cpp
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -o $@ $^
clean:
	rm run $(OBJDIR)/* 

.PHONY:clean

