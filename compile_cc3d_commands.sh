# Following cc3d compilation for MacOS: https://compucell3ddevelopersmanual.readthedocs.io/en/master/building_core_cc3d_cpp_code_mac.html

# Original command with errors
# [  0%] Building CXX object core/Logger/CMakeFiles/LoggerShared.dir/CC3DLogger.cpp.o
# In file included from /Users/svaren/src-cc3d/CompuCell3D/CompuCell3D/core/Logger/CC3DLogger.cpp:8:
# In file included from /Users/svaren/src-cc3d/CompuCell3D/CompuCell3D/core/Logger/CC3DLogger.h:19:
# In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/sstream:323:
# In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/__ostream/basic_ostream.h:19:
# In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/__locale_dir/num.h:12:
# In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/__algorithm/find.h:16:
# /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/__bit/countr.h:28:10: error: use of undeclared identifier '__builtin_ctzg'
#    28 |   return __builtin_ctzg(__t, numeric_limits<_Tp>::digits);
#       |          ^
# /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/__algorithm/sort.h:362:39: note: in instantiation of function template specialization 'std::__countr_zero<unsigned long long>' requested here
#   362 |     difference_type __tz_left  = std::__countr_zero(__left_bitset);
#       |   
# ...

# cmake -S ~/src-cc3d/CompuCell3D/CompuCell3D \
#     -B ~/src-cc3d/CompuCell3D_build \
#     -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python \
#     -DNO_OPENCL=ON  \
#     -DBUILD_STANDALONE=OFF \
#     -G "Unix Makefiles" \
#     -DCMAKE_INSTALL_PREFIX=~/src-cc3d/CompuCell3D_install

# Force to use Apple compiler - FIXED THE ISSUE
cmake -S ~/src-cc3d/CompuCell3D/CompuCell3D \
    -B ~/src-cc3d/CompuCell3D_build \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path) \
    -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DNO_OPENCL=ON \
    -DBUILD_STANDALONE=OFF \
    -G "Unix Makefiles" \
    -DCMAKE_INSTALL_PREFIX=~/src-cc3d/CompuCell3D_install \
    -Wno-dev

# [100%] Linking CXX shared module _SerializerDEPy.so
# [100%] Built target SerializerDEPy
# 72 warnings generated.
# [100%] Linking CXX shared module _PlayerPython.so
# [100%] Built target PlayerPythonNew
# 56 warnings generated.
# [100%] Linking CXX shared module _CompuCell.so
# [100%] Built target CompuCell

# Install
make install

# Test
python -m cc3d.run_script -i ~/src-cc3d/CompuCell3D/CompuCell3D/core/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d
# XML is valid!
# INFO: Random number generator: MersenneTwister
# INFO: 

# ------------------PERFORMANCE REPORT:----------------------
# -----------------------------------------------------------
# TOTAL RUNTIME 5 s : 198 ms = 5.198 s
# -----------------------------------------------------------
# -----------------------------------------------------------
# PYTHON STEPPABLE RUNTIMES
#             cellsort_2DSteppable:        0.00 ( 0.1%)
# -----------------------------------------------------------
#             Total Steppable Time:        0.00 ( 0.1%)
#     Compiled Code (C++) Run Time:        4.96 (95.4%)
#                       Other Time:        0.23 ( 4.5%)
# -----------------------------------------------------------