#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include "dbt.h"

static inline void do_op(const void* __restrict__ b1, const void* __restrict__ b2,
                         const void* __restrict__ target, size_t msglen) {
    int *i1 = (int*)b1;
    int *i2 = (int*)b2;
    int *t  = (int*)target;
    int count = msglen/sizeof(int);
    int i;
#pragma omp simd    
    for (i=0; i<count; i++) {
        t[i] = i1[i] + i2[i];
    }
}

static inline void do_op3(const void* __restrict__ b1, const void* __restrict__ b2,
                          const void* __restrict__ b3,
                         const void* __restrict__ target, size_t msglen) {
    int *i1 = (int*)b1;
    int *i2 = (int*)b2;
    int *i3 = (int*)b3;
    int *t  = (int*)target;
    int count = msglen/sizeof(int);
    int i;
#pragma omp simd    
    for (i=0; i<count; i++) {
        t[i] = i1[i] + i2[i] + i3[i];
    }
}

static void do_reduce(dbt_t db, void *sbuf, void *rbuf, size_t len, int n_frags) {
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
    static void *tmp = NULL;
    size_t scratch_size = (1<<17);
    if (scratch_size < chunk) {
        fprintf(stderr,"Scratch too small\n");
        return ;
    }
    if (NULL == tmp) {
        tmp = malloc(4*scratch_size);
    }
    void **scratch[4];
    for (i=0; i<4; i++) {
        scratch[i] = (void*)((char*)tmp + scratch_size*i);
    }
    
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

    assert(r_exp[0] == 0 || r_exp[1] == 0 || db.is_root);
    i = 0;
    int nrecv = 0;
    int need_reduce;
    while (rstep[0] < r_exp[0] || rstep[1] < r_exp[1] ||
           sstep[0] < s_exp[0] || sstep[1] < s_exp[1]) {
        active = 0;
        color = (i) % 2;
        rtree = db.p_t[color];
        stree = db.c_t[color];
        need_reduce = 0;
        if ((db.p[color] != -1) && (rstep[rtree] < r_exp[rtree]) && ((i/2) >= (db.h[rtree] - 1))) {
            /* offset = rtree*len/2 + rstep[rtree]*chunk; */
            MPI_Irecv(scratch[color], chunk, MPI_BYTE, db.p[color], 123,
                      MPI_COMM_WORLD, &reqs[active]);
#if DBG             > 1
            printf("Recv from %d, i %d tree %d, color %d\n", db.p[color], i, rtree, color);
#endif
            need_reduce = 1;
            active++;
            /* rstep[rtree]++; */
        }

        if ((db.c[color] != -1) && (sstep[stree] < s_exp[stree]) && ((i/2) >= db.h[stree])) {
            assert(n_children[stree]  == 1);
            offset = stree*len/2 + sstep[stree]*chunk;
            void *_sbuf;
            if (db.h[stree] == 0) {
                _sbuf = (char*)sbuf + offset;
            } else {
                _sbuf = scratch[2 + (sstep[stree] % 2)];
            }
            MPI_Isend(_sbuf, chunk, MPI_BYTE, db.c[color], 123,
                      MPI_COMM_WORLD, &reqs[active]);
#if DBG            > 1
            printf("Send to %d, soffset %d i %d tree %d, v[0] = %d, scratch_id %d\n",
                   db.c[color], offset, i, stree, ((int*)_sbuf)[0], 2 + sstep[stree] % 2);
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
        if (need_reduce) {
            void *_s1, *_s2, *_s3, *_t;
            offset = rtree*len/2 + (rstep[rtree]/n_parents[rtree])*chunk;
            
            if (db.is_root) {
                _s1 = (char*)sbuf + offset;
                _s2 = scratch[color];
                _t  = (char*)rbuf + offset;;
                do_op(_s1, _s2, _t, chunk);                
            } else {
                nrecv++;
                if (nrecv == n_parents[rtree]) {
                    assert(n_parents[rtree]  >= 0);

                    int scratch_id = -1;

                    scratch_id = 2 + ((rstep[rtree] / n_parents[rtree]) % 2);
                    _t = scratch[scratch_id];
                    _s2 = scratch[color];
                    _s1 = (void*)((char*)sbuf + offset);
                    if (nrecv == 2) {
                        _s3 = scratch[(color + 1) % 2];
#if DBG > 1
                    printf("Reduce3 v1 %d v2 %d, v3 %d color %d, scratch id %d, nrecv %d\n",
                           ((int*)(_s1))[0], ((int*)_s2)[0],((int*)_s3)[0], color, scratch_id, nrecv);
#endif
                        do_op3(_s1, _s2, _s3, _t, chunk);                
                    } else {
                        do_op(_s1, _s2, _t, chunk);                
                    }


                    nrecv = 0;
                }
            }
            rstep[rtree]++;
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
    dbt_init_reduce(size, rank, root, &dbt);

    int *sbuf = malloc(count2*sizeof(int));
    int *rbuf = malloc(count2*sizeof(int));
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

    for (i=0; i<count2; i++) {
        sbuf[i] = rank + i + 1;
    }

    int c, j;
    int status = 0, status_global = 0;
    int nf;
    for (c = count1; c<=count2; c*=2) {
        if (n_frags > 0) {
            nf = n_frags;
        } else {
            size_t fs = 131072;
            nf = (int)(c*sizeof(int)/fs);
        }
        if (nf  < 8) {
            nf = 8;
        }
        
        for (i=0; i<warmup; i++) {
            if (root == rank) {
                memset(rbuf, 0, c*sizeof(int));
            }
            do_reduce(dbt, sbuf, rbuf, c*sizeof(int), nf);
            if (do_check) {
                if (rank == root) {
                    for (j=0; j<c; j++) {
                        if (rbuf[j] != (size+1)*size/2 + j*size) {
                            fprintf(stderr, "Error: pos %d value %d expected %d\n", j, rbuf[j],
                                    (size+1)*size/2 + j*size);
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
            do_reduce(dbt, sbuf, rbuf, c*sizeof(int), nf);
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
    free(sbuf);
    free(rbuf);
    MPI_Finalize();
    return 0;
}
