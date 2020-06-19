SOURCES := dbt_bcast.c dbt_reduce.c dbt_test.c dbt_compute.c

OBJS := $(SOURCES:.c=.o)
CC=mpicc
CFLAGS=-g -O0
# CFLAGS=-O3 -fopenmp

all: dbt

dbt: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o dbt $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

clean:
	$(RM) $(OBJS) dbt
