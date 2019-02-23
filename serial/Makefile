
CXXFLAGS = -std=c++11
run_cmd=
PROG=Serial
ARGS=

# The C++ compiler
ifneq (,$(shell which icc 2>/dev/null))
CXX=icc
endif # icc

ifeq ($(shell uname), Linux)
LIBS+= -lrt 
endif

all:	release

release: *.cpp
	$(CXX) -O2 $(CXXFLAGS) -o $(PROG) $^ $(LIBS)

debug: *.cpp
	$(CXX) -O0 -g $(CXXFLAGS) -o $(PROG) $^ $(LIBS)

clean:
	$(RM) $(PROG) *.o *.d

test:
	$(run_cmd) ./$(PROG) $(ARGS)
