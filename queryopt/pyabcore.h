#include <iostream>
#include "kcore.h"
#include "bigraph.h"
#include <math.h>

using namespace std;

class Pyabcore{
    public:
        vector<bool> left; vector<bool> right;
        vector<vector<bicore_index_block*>> bicore_index_u; vector<vector<bicore_index_block*>> bicore_index_v;
        string dir;
        BiGraph g;
        int n1;
        int n2;
        Pyabcore(string dir);
        Pyabcore(int init_n1, int init_n2);
        void index();
        void index(int * inputA2, int D1, int D2);
        void query(int a, int b);
        vector<bool> get_left();
        vector<bool> get_right();
};