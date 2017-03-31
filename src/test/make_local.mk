LOC_SRC_DIR=$(PROJECT_ROOT)/test
EXT_FOLDER=$(filter-out doxygen/, $(filter-out test/, $(filter-out db/, \
	$(filter-out lib/, $(filter %/,$(shell ls -F ../))))))
EXT_INC_DIR=$(addprefix $(PROJECT_ROOT)/,$(addsuffix inc,$(EXT_FOLDER)))
EXT_INC_FLAGS=$(addprefix -I,$(EXT_INC_DIR))
EXT_OBJS=$(addprefix $(PROJECT_ROOT)/,$(addsuffix obj/*,$(EXT_FOLDER)))

CPPFLAGS+= $(EXT_INC_FLAGS)

LOC_SRCS=$(notdir $(wildcard $(LOC_SRC_DIR)/*.cpp))
TARGETS=$(LOC_SRCS:.cpp=)
