cd '~/Documents/self-driving-car/apollo5_svl/apollo-5.0'
./docker/scripts/dev_start.sh
./docker/scripts/dev_into.sh
./apollo.sh build_opt_gpu
bootstrap.sh
bridge.sh
