#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include "dbt.h"

void dbt_allreduce(dbt_t db, void *sbuf, void *rbuf, size_t len, int n_frags) {
    dbt_reduce(db, sbuf, rbuf, len, n_frags);
    dbt_bcast(db, rbuf, len, n_frags);
}
