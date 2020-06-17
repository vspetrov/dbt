#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include "dbt.h"

static void do_bcast(dbt_t db, void *buf, size_t len, int n_frags) {
    uint32_t i;
    int n_steps = n_frags*2;
    size_t chunk = len/n_steps;
    MPI_Request reqs[2];
    int active;
    int rstep[2] = {0 , 0};
    int sstep[2] = {0 , 0};
    uint32_t color;
    int stree, rtree;
    ptrdiff_t offset;
#if DBG > 1
    if (db.is_root) {
        printf("Bcast, len %zd, frags %d, steps %d, chunk %zd\n", len, n_frags, n_steps, chunk);
    }
#endif    
    int rank;
    int n_children[2] = {0, 0};
    int n_parents[2]  = {0, 0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (i=0; i<2; i++) {
        if (db.c[i] != -1) {
            n_children[db.c_t[i]]++;
        }
        if (db.p[i] != -1) {
            n_parents[db.p_t[i]]++;
        }
    }

    int r_exp[2] = {n_parents[0]*n_frags, n_parents[1]*n_frags};
    int s_exp[2] = {n_children[0]*n_frags, n_children[1]*n_frags};
    
    i = 0;
    while (rstep[0] < r_exp[0] || rstep[1] < r_exp[1] ||
           sstep[0] < s_exp[0] || sstep[1] < s_exp[1]) {
        active = 0;
        color = (i) % 2;
        rtree = db.p_t[color];
        stree = db.c_t[color];        
        if ((db.p[color] != -1) && (rstep[rtree] < r_exp[rtree]) && ((i/2) >= (db.h[rtree] - 1))) {
            offset = rtree*len/2 + rstep[rtree]*chunk;
            MPI_Irecv((char*)buf + offset, chunk, MPI_BYTE, db.p[color], 123,
                      MPI_COMM_WORLD, &reqs[active]);
#if DBG             > 1
            printf("Recv from %d, roffset %d i %d tree %d\n", db.p[color], offset, i, rtree);
#endif            
            active++;
            rstep[rtree]++;
        }

        if ((db.c[color] != -1) && (sstep[stree] < s_exp[stree]) && ((i/2) >= db.h[stree])) {
            assert(n_children[stree] > 0);
            offset = stree*len/2 + (sstep[stree]/n_children[stree])*chunk;
            MPI_Isend((char*)buf + offset, chunk, MPI_BYTE, db.c[color], 123,
                      MPI_COMM_WORLD, &reqs[active]);
#if DBG            > 1
            printf("Send to %d, soffset %d i %d tree %d\n", db.c[color], offset, i, stree);
#endif            
            active++;
            sstep[stree]++;
        }
        if (active) {
            MPI_Waitall(active, reqs, MPI_STATUS_IGNORE);
#if DBG            > 1
            printf("Completed %d requests, rstep %d:%d, sstep %d:%d\n", active,
                   rstep[0], rstep[1], sstep[0], sstep[1]);
#endif            
        }
        i++;
    }
#if DBG > 1
    printf("All done\n");
#endif
}

int main (int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char *var;
    int root = 0, i;
    size_t m1 = 1024;
    size_t m2 = (1<<19);

    int n_frags = -1;
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
    
    int count1 = m1/sizeof(int);
    int count2 = m2/sizeof(int);
    int iters = 1000;
    int warmup = 100;
    dbt_t dbt;
    dbt_init(size, rank, root, &dbt);

    int *buf = calloc(count2, sizeof(int));
    int do_check = 0;

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
    if (rank == root) {
        for (i=0; i<count2; i++) {
            buf[i] = 0xdeadbeef + i;
        }
    }
    int c, j;
    int status = 0, status_global;
    int nf;
    for (c = count1; c<=count2; c*=2) {
        if (n_frags > 0) {
            nf = n_frags;
        } else {
            size_t fs = 262144;
            nf = (int)(c*sizeof(int)/fs);
        }
        for (i=0; i<warmup; i++) {
            if (rank != root) {
                memset(buf, 0, c*sizeof(int));
            }
            do_bcast(dbt, buf, c*sizeof(int), nf);
            if (do_check) {
                if (rank != root) {
                    for (j=0; j<c; j++) {
                        if (buf[j] != 0xdeadbeef + j) {
                            fprintf(stderr, "Error: pos %d value %d expected %d\n", j, buf[j], 0xdeadbeef + j);
                            status = 1;
                            break;
                        }
                    }
                }
                MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if (status_global > 0) {
                    goto cleanup;
                }
            }
        }
        double total = 0;
        for (i=0; i<iters; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();            
            do_bcast(dbt, buf, c*sizeof(int), nf);
            total += MPI_Wtime() - t1;
        }
        total /= iters;
        double gmax, gmin, gavg;
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
    free(buf);
    MPI_Finalize();
    return 0;
}
