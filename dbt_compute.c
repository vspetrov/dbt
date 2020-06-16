/* #define MODE 1 */
#ifdef MODE
#include <iostream>
#include <string>
#include <sstream>
using namespace std;
#endif
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
        return v < vsize ? v : vsize - 1;
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
