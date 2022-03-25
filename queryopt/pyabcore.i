%module pyabcore
%{
#define SWIG_FILE_WITH_INIT
#include "pyabcore.h"
%}
%include "stl.i"
%template(BoolVector) std::vector<bool>;

%include "numpy.i"
%init %{
import_array();
%}

%apply (int* IN_ARRAY2,int DIM1,int DIM2) {(int* inputA2, int D1, int D2)}

%include "pyabcore.h"