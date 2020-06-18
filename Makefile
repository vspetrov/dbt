SOURCES := dbt_bcast.c dbt_reduce.c dbt_compute.c

OBJS := $(SOURCES:.c=.o)
CC=mpicc
# CFLAGS=-g -O0
CFLAGS=-O3 -fopenmp

all: dbt_bcast dbt_reduce

dbt_bcast: $(OBJS)
	$(CC) $(CFLAGS) -o dbt_bcast dbt_bcast.o dbt_compute.o $(LFLAGS) $(LIBS)

dbt_reduce: $(OBJS)
	$(CC) $(CFLAGS) -o dbt_reduce dbt_reduce.o dbt_compute.o $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

clean:
	$(RM) $(OBJS) dbt_reduce dbt_bcast
