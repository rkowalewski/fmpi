MPICXX ?= mpic++
CXX ?= g++

INCLUDES= -Iinclude -Iexternal/MPISynchronizedBarrier/include
CXXFLAGS =  -std=c++14 -O3 $(INCLUDES) -DNDEBUG -Wall

.PHONY: clean all

SRC_PATH := src
COMMON_DEPS := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))

all: build/factor_mpi.x build/factor_serial.x

build/factor_mpi.x: external/MPISynchronizedBarrier/lib/libmpisyncbarrier.a
	@mkdir -p build
	$(MPICXX) $(CXXFLAGS) -o $@ src/factor_mpi.cc $^

build/factor_serial.x:
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ src/factor_serial.cc $^

external/MPISynchronizedBarrier/lib/libmpisyncbarrier.a:
	make -C external/MPISynchronizedBarrier clean rebuild

clean:
	rm -f build/*.x
