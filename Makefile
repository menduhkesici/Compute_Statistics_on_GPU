###########################################################

## CUDA C/C++ PROJECT FILE STRUCTURE ##

#|--> Project/
#	|--> Makefile
#	|--> main.cpp
#	|--> src/		(source files)
#		|--> file1.cpp
#		|--> file2.cpp
#		|--> file_cuda1.cu
#		|--> file_cuda2.cu
#	|--> include/	(header files)
#		|--> file1.h
#		|--> file2.h
#		|--> file1.cuh
#		|--> file2.cuh
#	|--> bin/		(object files and dependency files)
#	|--> Project	(executable output)

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

##########################################################

## COMPILE OPTIONS ##

# Linker options:
LD= g++
LD_FLAGS= -std=c++17 -O3
LD_LIBS= 

# C++ compiler options:
CXX= g++
CXX_FLAGS= -MMD -std=c++17 -O3
CXX_LIBS=

# C compiler options:
CC= gcc
CC_FLAGS= -MMD -O3
CC_LIBS=

# NVCC compiler options:
NVCC= nvcc
NVCC_FLAGS= -MMD -std=c++17 -O3 -Xptxas -O3
NVCC_LIBS=

##########################################################

## CUDA DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR= /usr/local/cuda-11.7/targets/x86_64-linux

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib

# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include

# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart
# -lcufft

##########################################################

## MAKE VARIABLES ##

# Target executable name:
EXE := $(shell basename $(CURDIR))

# Object files:
OBJS := $(OBJ_DIR)/main.cpp.o\
 $(patsubst $(SRC_DIR)/%, $(OBJ_DIR)/%.o, $(shell find $(SRC_DIR) -name '*.cpp' -or -name '*.c' -or -name '*.cu'))

# Dependencies:
DEPS := $(OBJS:%.o=%.d)

##########################################################

## COMPILE ##

# Link compiled object files to target executable
$(EXE) : $(OBJS)
	$(LD) $(LD_FLAGS) $(OBJS) -o $@ $(LD_LIBS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
	@echo "-- Build completed --"

# Include dependency files
-include $(DEPS)

# Compile main.cpp file to object file
$(OBJ_DIR)/main.cpp.o : main.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(CXX_LIBS)

# Compile C++ source files to object files
$(OBJ_DIR)/%.cpp.o : $(SRC_DIR)/%.cpp 
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(CXX_LIBS)

# Compile C source files to object files
$(OBJ_DIR)/%.c.o : $(SRC_DIR)/%.c 
	@mkdir -p $(@D)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files
$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Run executable after build
.PHONY: run
run: $(EXE)
	@echo "-- Program is started --"
	@./$(EXE)
	@echo "-- Program terminated successfully --"

# Profile the application
.PHONY: profile
profile: $(EXE)
	nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-device=all ./$(EXE) -v

# Open last profiling report
.PHONY: open_last_report
open_last_report:
	find report*.nsys-rep -printf '%T+ %p\n' | sort -r | head -n 1 | xargs -r nsys-ui

# Delete old profiling reports and rename last report to report1
.PHONY: delete_old_reports
delete_old_reports:
	find report*.nsys-rep -printf '%T+ %p\n' | sort -r | tail -n +2 | xargs -r rm -f
	find report*.nsys-rep | xargs -r -i{}  mv {} report1.nsys-rep

# Clean output files
.PHONY: clean
clean:
	rm -f $(EXE) *.nsys-rep
	rm -rf $(OBJ_DIR)

###########################################################
