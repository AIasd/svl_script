cd '/home/zhongzzy9/Documents/self-driving-car/apollo'
./docker/scripts/dev_start.sh
./docker/scripts/dev_into.sh
./apollo.sh build_opt_gpu
./scripts/bootstrap_lgsvl.sh
./scripts/bridge.sh
