#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include "dbt.h"

enum {
    TEST_BCAST,
    TEST_REDUCE,
    TEST_ALLREDUCE,
};

static int check_bcast(int *buf, int count, int is_root, int root) {
    int status = 0;
    int j;
    if (!is_root) {
        for (j=0; j<count; j++) {
            if (buf[j] != root +1 + j) {
                fprintf(stderr, "Error: pos %d value %d expected %d\n",
                        j, buf[j], root + 1 + j);
                status = 1;
                break;
            }
        }
    }
    return status;
}

static int check_reduce(int *buf, int count, int is_root, int size) {
    int status = 0;
    int j;
    if (is_root) {
        for (j=0; j<count; j++) {
            if (buf[j] != (size+1)*size/2 + j*size) {
                fprintf(stderr, "Error: pos %d value %d expected %d\n", j, buf[j],
                        (size+1)*size/2 + j*size);
                status = 1;
                break;
            }
        }
        
    }
    return status;
}

int main (int argc, char **argv) {
    int    root    = 0;
    int    status = 0;
    int    status_global = 0;    
    size_t m1      = 1024;
    size_t m2      = (1<<19);
    int    n_frags = -1;
    int    rank, size, i, count1, count2, test;
    int    iters, warmup, do_check, c, j, nf;
    int    *sbuf, *rbuf;
    char   *var;
    dbt_t  dbt;
    double gmax, gmin, gavg, total, t1;
    size_t fs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    test     = TEST_BCAST;
    iters    = 1000;
    warmup   = 100;
    do_check = 0;
    
    var = getenv("DBT_ROOT");
    if (var) {
        root = atoi(var);
    }
    var = getenv("DBT_M1");
    if (var) {
        m1 = atoi(var);
    }
    var = getenv("DBT_M2");
    if (var) {
        m2 = atoi(var);
    }
    var = getenv("DBT_NFRAGS");
    if (var) {
        n_frags = atoi(var);
    }
    var = getenv("DBT_ITERS");
    if (var) {
        iters = atoi(var);
    }
    var = getenv("DBT_WARMUP");
    if (var) {
        warmup = atoi(var);
    }
    var = getenv("DBT_CHECK");
    if (var) {
        do_check = atoi(var);
    }
    var = getenv("DBT_TEST");
    if (var) {
        if (0 == strcmp(var, "bcast")) {
            test = TEST_BCAST;
        }
        if (0 == strcmp(var, "reduce")) {
            test = TEST_REDUCE;
        }
        if (0 == strcmp(var, "allreduce")) {
            test = TEST_ALLREDUCE;
        }
    }
    
    count1 = m1/sizeof(int);
    count2 = m2/sizeof(int);
    sbuf = malloc(count2*sizeof(int));
    rbuf = malloc(count2*sizeof(int));


    dbt_init(size, rank, root, &dbt);

    for (i=0; i<count2; i++) {
        sbuf[i] = rank + i + 1;
    }

    for (c = count1; c<=count2; c*=2) {
        if (n_frags > 0) {
            nf = n_frags;
        } else {
            fs = 131072;
            nf = (int)(c*sizeof(int)/fs);
            if (nf  < 8) {
                nf = 8;
            }
        }
        
        for (i=0; i<warmup; i++) {
            if (root == rank) {
                memset(rbuf, 0, c*sizeof(int));
            }
            if (test == TEST_BCAST) {
                dbt_bcast(dbt, dbt.is_root ? sbuf : rbuf, c*sizeof(int), nf );
            } else if (test == TEST_REDUCE) {
                dbt_reduce(dbt, sbuf, rbuf, c*sizeof(int), nf);
            }
            
            if (do_check) {
                if (test == TEST_BCAST) {
                    status = check_bcast(rbuf, c, dbt.is_root, root);
                } else if (test == TEST_REDUCE) {
                    status = check_reduce(rbuf, c, dbt.is_root, size);
                }
                MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if (status_global > 0) {
                    goto cleanup;
                }
            }
        }
        total = 0;
        for (i=0; i<iters; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();            
            if (test == TEST_BCAST) {
                dbt_bcast(dbt, dbt.is_root ? sbuf : rbuf, c*sizeof(int), nf );
            } else if (test == TEST_REDUCE) {
                dbt_reduce(dbt, sbuf, rbuf, c*sizeof(int), nf);
            }
            total += MPI_Wtime() - t1;
        }
        total /= iters;
        MPI_Reduce(&total, &gmax, 1 ,MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total, &gmin, 1 ,MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total, &gavg, 1 ,MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (0 == rank) {
            gmax *= 1e6;
            gmin *= 1e6;
            gavg *= 1e6/size;
            printf("%12zd\t%8.1f\t%8.1f\t%8.1f\n",c*sizeof(int), gmax, gmin, gavg);
        }
    }
cleanup:
    free(sbuf);
    free(rbuf);
    MPI_Finalize();
    return 0;
}
