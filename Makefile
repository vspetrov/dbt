SOURCES := dbt_bcast.c dbt_compute.c

OBJS := $(SOURCES:.c=.o)
CC=mpicc

all: dbt_bcast

dbt_bcast: $(OBJS)
	$(CC) $(CFLAGS) -o dbt_bacst dbt_bcast.o dbt_compute.o $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<
