#ifndef DBT_H
#define DBT_H
#define DBG 2
enum {
    DBT_T1,
    DBT_T2,
};

typedef struct double_bin_tree {
    int is_root;
    int p[2];
    int p_t[2];
    int c[2];
    int c_t[2];
    int h[2];
    int max_h;
} dbt_t;
void dbt_init(int size, int rank, int root, dbt_t *dbt);
void dbt_init_reduce(int size, int rank, int root, dbt_t *dbt);
void dbt_reduce(dbt_t db, void *sbuf, void *rbuf, size_t len, int n_frags);
void dbt_allreduce(dbt_t db, void *sbuf, void *rbuf, size_t len, int n_frags);
void dbt_bcast(dbt_t db, void *buf, size_t len, int n_frags);
#endif
