LOC_PATH=$(PROJECT_ROOT)/beamline
LOC_INC_DIR=$(LOC_PATH)/inc
LOC_SRC_DIR=$(LOC_PATH)/src
LOC_OBJ_DIR=$(LOC_PATH)/obj

EXT_FOLDER=util pywrapper
EXT_INC_DIR=$(addprefix $(PROJECT_ROOT)/,$(addsuffix /inc,$(EXT_FOLDER)))
EXT_INC_FLAGS=$(addprefix -I,$(EXT_INC_DIR))

LOCFLAGS= -I$(LOC_INC_DIR) $(EXT_INC_FLAGS)

LOC_SRCS=$(notdir $(wildcard $(LOC_SRC_DIR)/*.cpp))
LOC_OBJS=$(addprefix $(LOC_OBJ_DIR)/,$(LOC_SRCS:.cpp=.o))
LOC_CUSRCS:=$(notdir $(wildcard $(LOC_SRC_DIR)/*.cu))
LOC_CUOBJS=$(addprefix ./obj/,$(LOC_CUSRCS:.cu=.cu.o))
