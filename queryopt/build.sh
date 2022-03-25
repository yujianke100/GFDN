swig -c\+\+ -python pyabcore.i
python setup.py build_ext --inplace
python test.py
cp ./_pyabcore* pyabcore.py ../