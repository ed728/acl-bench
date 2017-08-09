CC = g++
INC = -I $(ACL_ROOT)/ -I $(ACL_ROOT)/include
LIBS = -L $(ACL_ROOT)/build -larm_compute -lOpenCL
SRC = main.cpp TestTensor.cpp
CFLAGS = -std=c++14

acl_bench: $(SRC)
	$(CC) $(INC) $(CFLAGS) $(LIBS) $^ -o $@
