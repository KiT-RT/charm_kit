cd KiT-RT
git checkout master
git pull origin
cd tools/singularity
singularity exec kit_rt.sif ./install_kitrt_singularity.sh
cd ../../../