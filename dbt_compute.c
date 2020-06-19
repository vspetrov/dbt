/* #define MODE 1 */
#ifdef MODE
#include <iostream>
#include <string>
#include <sstream>
using namespace std;
#endif
#include <mpi.h>
#include "dbt.h"
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

static int get_root(int vsize) {
    int r = 1;
    while (r <= vsize) r *= 2;
    return r/2 - 1;
}

static int get_height(int vrank) {
    int h = 1;
    if (vrank % 2 == 0) return 0;
    vrank++;
    while ((vrank & (1 << h)) == 0) h++;
    return h;
}

static int get_left_child(int vrank, int height) {
    return height == 0 ? -1 : vrank - (1 << (height - 1));
}

static int get_right_child(int vsize, int vrank, int height, int troot) {
    if (vrank == vsize - 1 || height == 0) {
        return -1;
    }else if (vrank == troot) {
        return vrank + get_root(vsize - troot - 1) + 1;
    } else {
        int v = vrank + (1 << (height - 1));
        return v < vsize ? v : vrank + get_root(vsize - vrank - 1) + 1;
    }
}

static void get_children(int vsize, int vrank, int height, int troot, int *c) {
    c[0] = get_left_child(vrank, height);
    c[1] = get_right_child(vsize, vrank, height, troot);
}

static int get_parent(int vsize, int vrank, int height, int troot) {
    if (vrank == troot) {
        return -1;
    } else if (height == 0) {
        return ((vrank/2) % 2 == 0) ? vrank + 1 : vrank - 1;        
    } else {
        vrank++;
        if ((((1<<(height+1)) & vrank) > 0) || (vrank + (1<<height)) > vsize) {
            return vrank - (1<<height) - 1;
        } else {
            return vrank + (1<<height) - 1;
        }
    }
}

void dbt_compute(int rank, int size, int *height, int *parent, int *children) {
    int troot = get_root(size);
    *height = get_height(rank);
    get_children(size, rank, *height, troot, children);
    *parent = get_parent(size, rank, *height, troot);
}

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

void dbt_init(int size, int rank, int root, dbt_t *dbt) {
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
    if (0 == rank) {
        printf("\n\n");fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif    
    db.is_root = (root == rank);
    db.max_h = max_h;
    *dbt = db;
}

#ifdef MODE
int main(int argc, char *argv[])
{
    int vsize, vrank;
    istringstream is;
    is.str(std::string(argv[1])+" " + std::string(argv[2]));
    is >> vsize >> vrank;
    int troot = get_root(vsize);
    std::cout << "vsize: " << vsize << " vrank: " << vrank << " vroot: " << troot << '\n';

    int parent;
    int children[2];

    int height = get_height(vrank);
    get_children(vsize, vrank, height, troot, children);
    parent = get_parent(vsize, vrank, height, troot);
    std::cout << "height: " << height << " children: " << children[0] << ", " <<
              children[1] << " parent: " << parent << '\n';
    return 0;
}
#endif
