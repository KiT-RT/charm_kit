# clone KiT-RT
git clone https://github.com/KiT-RT/kitrt_code.git
# go to KiT-RT directory
mv kitrt_code KiT-RT
cd KiT-RT

# checkout the branch new_radiation_test_cases
git fetch origin
git checkout master
# load all submodules
git submodule update --init --recursive

# create build directory
mkdir build_singularity

# navigate to directory where the singularity .def script is located
cd tools/singularity

# build the singularity container. This requires root privileges
echo "sudo access needed to build the singularity container. This will hopefully not be neccessary in the future."
sudo sh build_container.sh

# compile KiT-RT within the singularity container
chmod +x install_kitrt_singularity.sh
singularity exec kit_rt.sif ./install_kitrt_singularity.sh

# go back to CharmNet repo
cd ../../../