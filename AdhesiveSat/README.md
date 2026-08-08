# Levi commands to compile
```
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

cd ~/src-cc3d/CompuCell3D_build
make -j 8

make install
```
