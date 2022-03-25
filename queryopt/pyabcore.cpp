#include "pyabcore.h"
Pyabcore::Pyabcore(string dir){
    this->dir = dir;
}
Pyabcore::Pyabcore(int init_n1, int init_n2){
    n1 = init_n1;
    n2 = init_n2;
}
void Pyabcore::index(){
    cout << "start ComShrDecom in :" << dir << endl;
    
    BiGraph g(dir);
    coreIndexKCore(g);
    build_bicore_index(g, bicore_index_u, bicore_index_v);
    left.resize(g.num_v1, false); right.resize(g.num_v2, false);
    cout << "finished!" << endl;
}
void Pyabcore::index(int* inputA2, int D1, int D2){
    cout << "start ComShrDecom" << endl;
    
    BiGraph g(inputA2, D1, D2, n1, n2);
    coreIndexKCore(g);
    build_bicore_index(g, bicore_index_u, bicore_index_v);
    left.resize(g.num_v1, false); right.resize(g.num_v2, false);
    cout << "finished!" << endl;
}
void Pyabcore::query(int a, int b){
    cout << "start query" << endl;
    retrieve_via_bicore_index(g, bicore_index_u, bicore_index_v, left, right, a, b);
    cout << "finished!" << endl;
}

vector<bool> Pyabcore::get_left(){
    return left;
}
vector<bool> Pyabcore::get_right(){
    return right;
}