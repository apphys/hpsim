CC=nvcc
CXX=nvcc
PROJECT_ROOT:=$(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

EPICS_INC_FLAGS=-I/ade/epics/supTop/base/R3.14.11/include/os/Linux -I/ade/epics/supTop/base/R3.14.11/include -D_POSIX_C_SOURCE=199506L -D_POSIX_THREADS -D_XOPEN_SOURCE=500 -D_X86_64_ -DUNIX -D_BSD_SOURCE -Dlinux -D_REENTRANT -m64 
EPICS_LD_FLAGS=-L/ade/epics/supTop/base/R3.14.11/lib/linux-x86_64 -lcas -lgdd -lasHost -ldbStaticHost -lregistryIoc -lca -lCom -lpthread -lreadline -lncurses -lm -lrt -ldl -lgcc
PYTHON_INC_FLAGS=-I/usr/include/python2.6
PYTHON_LD_FLAGS=-L./usr/lib/python2.6/config/ -lpython2.6 -lpthread
GL_LD_FLAGS+=-L./lib -lglut -lGLEW_x86_64 -lGLU

CPPFLAGS+=-arch=sm_35 -Xcompiler '-fPIC' -Xcompiler '-fopenmp' -O3 -g -w #-DDOUBLE_PRECISION #-D_DEBUG
CPPFLAGS+=$(EPICS_INC_FLAGS)
CPPFLAGS+=$(PYTHON_INC_FLAGS)
#CPPFLAGS+=-Xptxas -v -Xptxas -dlcm=ca

LDFLAGS=-lcurand -lsqlite3
