#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#define DBG 0
enum {
    DBT_T1,
    DBT_T2,
};

typedef struct b_tree {
    int p;
    int p_c;
    int c[2];
    int c_c[2];
} b_tree_t;
typedef struct double_bin_tree {
    int is_root;
    int p[2];
    int p_t[2];
    int c[2];
    int c_t[2];
    int h[2];
    b_tree_t t1;
    b_tree_t t2;
} dbt_t;

void dbt_compute(int rank, int size, int *height, int *parent, int *children);
#if 0
static inline void compute_tree(int vrank, int vsize, int *height_p, int *parent_p, int *children_p) {
    int height, parent, children[2];
    height = (0 == vrank % 2) ? 1 : __builtin_ffs(~vrank);

    assert(height >= 1);

    if (height == 1) {
        children[0] = children[1] = -1;
        parent = (0 == (vrank/2) % 2) ? vrank + 1 : vrank - 1;
    } else {
        int dist = 1 << (height-2);
        int pos  = vrank/(1 << height);
        assert(dist <= vrank);
        children[0] = vrank - dist;
        children[1] = vrank + dist < vsize ? vrank + dist: - 1;
        dist = 1 << (height-1);
        parent = (0 == pos % 2) ? vrank + dist: vrank - dist;
        if (parent >= vsize) parent = -1;
    }
    *height_p = height;
    *parent_p = parent;
    children_p[0] = children[0];
    children_p[1] = children[1]; 
}
#endif

static inline int inEdgeColor(int p, int i, int h, int t1_root) {
    int i1;
    if (i == t1_root) return 1;
    while ((i & (1 << h)) == 0) h++;
    if ((((1<<(h+1)) & i) > 0) || (i + (1<<h)) > p) {
        i1 = i - (1<<h);
    } else {
        i1 = i + (1<<h);
    }
   return inEdgeColor(p, i1, h, t1_root) ^ (((p/2) % 2) == 1) ^ (i1 > i);
   /* return inEdgeColor(p, i1, h, t1_root) ^ 0 ^ (i1 > i); */
}

static inline void computeT1Colors(int vrank, int vsize, int t1_root, int t1_height,
                                   const int *t1_children, int *c_parent,
                                   int *c_children) {
    int h;
    int c1_parent, c1_children[2];
    h = 1;
    assert(vrank % 2 == 1);
    c1_parent = inEdgeColor(vsize, vrank+1, h, t1_root+1);
    assert(t1_children[0] >= 0);
    /* Child 1*/
    int c = t1_children[0];
    if (t1_height > 1) {
        h = 1;
        assert(c % 2 == 1);
        c1_children[0] = inEdgeColor(vsize, c+1, h, t1_root+1);
    } else {
        assert(t1_height == 1);
        /* children 1 is leaf */
        h = 1;
        int t2_c = vsize - 1 - c;
        assert(t2_c % 2 == 1);
        c1_children[0] = inEdgeColor(vsize, t2_c+1, h, t1_root+1);
    }

    /* Child 2 */
    if (t1_children[1] >= 0) {
        c = t1_children[1];
        if (t1_height > 1) {
            h = 1;
            assert(c % 2 == 1);
            c1_children[1] = inEdgeColor(vsize, c+1, h, t1_root+1);
        } else {
            assert(t1_height == 1);
            /* children 1 is leaf */
            h = 1;
            int t2_c = vsize - 1 - c;
            assert(t2_c % 2 == 1);
            c1_children[1] = inEdgeColor(vsize, t2_c+1, h, t1_root+1);
        }
    }
    c_children[0] = c1_children[0];
    c_children[1] = c1_children[1];
    *c_parent = c1_parent;
}

static inline void print_dbt(int rank, dbt_t db) {
    char buf[512];
    sprintf(buf, "%3d: R [%3d(%d) : %3d(%d)]  B [%3d(%d) : %3d(%d)] H [%d %d]",
            rank, db.c[0], db.c_t[0], db.p[0], db.p_t[0],
            db.c[1], db.c_t[1], db.p[1], db.p_t[1], db.h[0], db.h[1]);
    printf("%s\n", buf);
}

#define VROOT INT_MAX

#define TO_ORIG(_v) (_v) = ((_v == INT_MAX) ? root : ((_v < root) ? _v : _v + 1))

static void init_dbt_t(int size, int rank, int root, dbt_t *dbt) {
    int vrank = rank < root ? rank : rank - 1;
    int vsize = size - 1;
    int extra = (vsize % 2) ? vsize - 1 : -1;
    int i;
    dbt_t db;
    int i_am_extra = (extra != -1 && extra == vrank);
    memset(&db, 0, sizeof(db));
    if (-1 != extra) {
        vsize--;
    }

    if (i_am_extra) {
        vrank = 0;
    }

    int max_h = 0;
    int t1_root = 1;
    while (t1_root * 2 <= vsize) {
        t1_root *= 2;
        max_h++;
    }
    t1_root--;
    int t2_root = vsize - 1 - t1_root;
    if (rank != root) {
        int t1_height, t1_parent, t1_children[2];
        int t2_height, t2_parent, t2_children[2];        

        /* T1 */
        dbt_compute(vrank, vsize, &t1_height, &t1_parent, t1_children);
        /* printf("vrank %d, t1_height %d, children: %d, %d, parent: %d\n", */
        /* vrank, t1_height, t1_children[0], t1_children[1], t1_parent); */

        /* T2 */
        int vrank_t2 = vsize - vrank - 1;
        int mirror_children[2];
        dbt_compute(vrank_t2, vsize, &t2_height, &t2_parent, mirror_children);
        t2_parent = t2_parent >=0? vsize - 1 - t2_parent: -1;
        t2_children[0] = mirror_children[1] >=0 ? vsize - 1 - mirror_children[1]: -1;
        t2_children[1] = mirror_children[0] >=0 ? vsize - 1 - mirror_children[0] : -1;
        /* printf("vrank %d, t2_height %d, children: %d, %d, parent: %d\n", */
        /* vrank, t2_height, t2_children[0], t2_children[1], t2_parent); */



        /* Coloring */
        int c1_parent, c1_children[2], c2_parent, c2_children[2], h;
        /* Compute incoming edge color in t1 - c1_parent */

        if (t1_height > 0) {
            /* Innder t1 nodes */
            assert(t2_children[0] == -1 && t2_children[1] == -1);
            computeT1Colors(vrank, vsize, t1_root, t1_height,
                            t1_children, &c1_parent, c1_children);
            c2_parent = 1 - c1_parent;
        } else {
            /* Inner t2 nodes */
            assert(t1_children[0] == -1 && t1_children[1] == -1);                
            int mirror_vrank = vsize - 1 -vrank;
            int mirror_children[2], mirror_colors[2];
            mirror_children[0] = t2_children[1] > -1? vsize - 1 - t2_children[1] : -1;
            mirror_children[1] = t2_children[0] > -1? vsize - 1 - t2_children[0] : -1;
            computeT1Colors(mirror_vrank, vsize, t1_root, t2_height,
                            mirror_children, &c1_parent, mirror_colors);
            c2_parent = 1 - c1_parent;
            c2_children[0] = 1 - mirror_colors[1];
            c2_children[1] = 1 - mirror_colors[0];
        }
        if (-1 != extra) {
            if (vrank == 0) {
                assert(t2_children[0] == -1 && t2_children[1] != -1);
                t2_children[0] = extra;
                c2_children[0] = 1 - c2_children[1];
            }
            if (vrank == vsize - 1) {
                assert(t1_children[1] == -1 && t1_children[0] != -1);
                t1_children[1] = extra;
                c1_children[1] = 1 - c1_children[0];
            }
        }
            
        if (vrank == t1_root) {
            assert(t1_parent == -1 && t2_parent != -1);
            c2_parent = 0;
            t1_parent = VROOT;
            c1_parent = 1;
        }
        if (vrank == t2_root) {
            assert(t2_parent == -1 && t1_parent != -1);
            c1_parent = 1;
            t2_parent = VROOT;
            c2_parent = 0;
        }
        db.p[0] = db.p[1] = -1;
        db.c[0] = db.c[1] = -1;
            
        if (t1_parent != -1) {
            db.p[c1_parent] = t1_parent;
            db.p_t[c1_parent] = DBT_T1;
        }
        if (t2_parent != -1) {
            assert(db.p[c2_parent] == -1);
            db.p[c2_parent] = t2_parent;
            db.p_t[c2_parent] = DBT_T2;
        }

        for (i=0; i<2; i++) {
            if (t1_children[i] != -1) {
                assert(db.c[c1_children[i]] == -1);
                db.c[c1_children[i]] = t1_children[i];
                db.c_t[c1_children[i]] = DBT_T1;
            }
        }
        for (i=0; i<2; i++) {
            if (t2_children[i] != -1) {
                assert(db.c[c2_children[i]] == -1);
                db.c[c2_children[i]] = t2_children[i];
                db.c_t[c2_children[i]] = DBT_T2;
            }
        }

        db.t1.p = t1_parent; db.t1.p_c = c1_parent;
        db.t1.c[0] = t1_children[0]; db.t1.c_c[0] = c1_children[0];
        db.t1.c[1] = t1_children[1]; db.t1.c_c[1] = c1_children[1];

        db.t2.p = t2_parent; db.t2.p_c = c2_parent;
        db.t2.c[0] = t2_children[0]; db.t2.c_c[0] = c2_children[0];
        db.t2.c[1] = t2_children[1]; db.t2.c_c[1] = c2_children[1];
        
        if (i_am_extra) {
            t1_children[0] = t1_children[1] = t2_children[0] = t2_children[1] = -1;
            t1_parent = vsize - 1;
            t2_parent = 0;
            c2_parent = c2_children[0];
            c1_parent = 1 - c2_parent;
            db.c[0] = db.c[1] = -1;
            
            db.p[c1_parent] = t1_parent;
            db.p_t[c1_parent] = DBT_T1;
            db.p[c2_parent] = t2_parent;
            db.p_t[c2_parent] = DBT_T2;
            vrank = extra;

            //TODO compute height
            db.t1.p = t1_parent; db.t1.p_c = c1_parent;
            db.t2.p = t2_parent; db.t2.p_c = c2_parent;

            db.t1.c[0] = db.t1.c[1] = db.t2.c[0] = db.t2.c[1] = -1;
            db.h[0] = db.h[1] = max_h + 1;
        } else {
            db.h[0] = max_h - t1_height + 1;
            db.h[1] = max_h - t2_height + 1;
        }
        
    } else {
        /* printf("t1_root %d, t2_root %d\n", t1_root, t2_root); */
        db.p[0] = db.p[1] = -1;
        db.c[0] = t2_root;
        db.c[1] = t1_root;
        db.c_t[0] = DBT_T2;
        db.c_t[1] = DBT_T1;
        db.t1.p = db.t2.p = -1;
        db.t1.c[0] = t1_root; db.t1.c_c[0] = 1; db.t1.c[1] = -1;
        db.t2.c[0] = t2_root; db.t2.c_c[0] = 0; db.t2.c[1] = -1;
        db.h[0] = db.h[1] = 0;
    }

    /* From vrank to original ranks */
    TO_ORIG(db.p[0]);
    TO_ORIG(db.p[1]);
    TO_ORIG(db.c[0]);
    TO_ORIG(db.c[1]);

#if DBG >= 1
    for (i=0 ; i<size; i++) {
        if (i == rank) {
            fflush(stdout);
            print_dbt(vrank, db);
            usleep(1000);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif    
    db.is_root = (root == rank);
    *dbt = db;
}



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
    init_dbt_t(size, rank, root, &dbt);

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
