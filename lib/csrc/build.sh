export CUDA_HOME="/usr/local/cuda"
cd dcn_v2
rm -r build/ _ext.cpython-3*
python setup.py build_ext --inplace

cd ../extreme_utils
rm -r build/ _ext.cpython-3*
python setup.py build_ext --inplace

cd ../roi_align_layer
rm -r build/ _ext.cpython-3*
python setup.py build_ext --inplace
