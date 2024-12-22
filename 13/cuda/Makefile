# equivalent, it seems
NVCC_CCBIN=/usr/bin/gcc-13
NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc

DEPDIR := .deps

all: libmycuda.a

run: all
	LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6/targets/x86_64-linux/lib/ cargo run -- ../input_mod.txt

rust: libmycuda.a
	cargo build

NVCCFLAGS=-ccbin=$(NVCC_CCBIN) --gpu-code sm_75 --gpu-architecture compute_75 -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -rdc=true --expt-relaxed-constexpr -O4 --extended-lambda
NVCCLINKFLAGS=-ccbin=$(NVCC_CCBIN) --gpu-code sm_75 --gpu-architecture compute_75 -O4 --extended-lambda -lcudart -lcudadevrt

NVLDLIBS=-lcudadevrt

NVPPFLAGS=-I/opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/include

CPPFLAGS=-I/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/include/

CXXFLAGS=-Wall -Wextra -std=c++20 -O3

# CFLAGS=-Wall -Wextra -O3 -std=c99

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# read_example_test: LDLIBS+=-lgtest_main -lgtest
# read_example_test: read_example_test.cpp.o helpers.cu.o helpers.cpp.o read.cpp.o
# 	echo $(DEPFILES)
# 	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

%.cu.o : %.cu Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(NVCC) $(NVCCFLAGS) $(NVPPFLAGS) $< -c -o $@ $(CPPFLAGS) $(DEPFLAGS)

# https://stackoverflow.com/a/22116121
%.cu.l.o : %.cu.o Makefile
	$(NVCC) $(NVCCLINKFLAGS) $< -dlink -o $@

%.cpp.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ -c $(DEPFLAGS)

%.c.o : %.c Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(NVCC_CCBIN) $(CFLAGS) $(CPPFLAGS) $< -o $@ -c $(DEPFLAGS)

libmycuda.a: c_interface.cpp.o claw.cu.l.o cuda_interface.cu.l.o claw.cu.o cuda_interface.cu.o
	ar rsc $@ $^


$(DEPDIR): ; @mkdir -p $@

CPPSRCS := $(wildcard *.cpp)
DEPFILES := $(CPPSRCS:%.cpp=$(DEPDIR)/%.d)
CUSRCS := $(wildcard *.cu)
DEPFILES += $(CUSRCS:%.cu=$(DEPDIR)/%.d)
CUSRCS := $(wildcard *.c)
DEPFILES += $(CUSRCS:%.c=$(DEPDIR)/%.d)
CUSRCS := $(wildcard *.cc)
DEPFILES += $(CUSRCS:%.cc=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

rust_clean:
	cargo clean

clean: rust_clean
	rm -f *_test main *.o libmycuda.a

.PHONY: clean all rust
