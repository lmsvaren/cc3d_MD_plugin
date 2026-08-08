# Setting up environment for MacOS
Directions from [CC3D Developers Manual](https://compucell3ddevelopersmanual.readthedocs.io/en/master/building_core_cc3d_cpp_code_mac.html)

### Ensure you have `xcode-select`
```
xcode-select --install
```

### Pull all GitHub repos needed
```
mkdir -p ~/src-cc3d
cd ~/src-cc3d
git clone https://github.com/CompuCell3D/CompuCell3D.git
git clone https://github.com/CompuCell3D/cc3d-player5.git
git clone https://github.com/CompuCell3D/cc3d-twedit5.git
```

### Create conda environment
Make sure environment.yaml is in `~/src-cc3d`
```
cd ~/src-cc3d
conda env create -f environment.yaml --name cc3d_compile
```

### Activate
After activating environment, you can compile cc3d
```
conda activate cc3d_compile
```
