#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include "dbt.h"


void dbt_bcast(dbt_t db, void *buf, size_t len, int n_frags) {
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
