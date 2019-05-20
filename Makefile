MPICXX ?= mpic++
CXX ?= g++

INCLUDES= -Iinclude -Iexternal/MPISynchronizedBarrier/include
#CXXFLAGS =  -std=c++14 -O3 $(INCLUDES) -Wall -DNDEBUG
CXXFLAGS =  -std=c++14 -O0 -ggdb3 $(INCLUDES) -Wall

.PHONY: clean all

SRC_PATH := src

#COMMON_PATH := $(SRC_PATH)/common
#COMMON_DEPS := $(foreach x, $(COMMON_PATH), $(wildcard $(addprefix $(x)/*,.c*)))

all: build/factor_mpi.x build/factor_serial.x

build/factor_mpi.x: src/factor_mpi.cc src/Benchmark.cc src/synchronized_barrier.cc
	@mkdir -p build
	$(MPICXX) $(CXXFLAGS) -o $@ $^

build/factor_serial.x:
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ src/factor_serial.cc $^


clean:
	rm -f build/*.x
