cd '/home/zhongzzy9/Documents/self-driving-car/apollo_master'
./docker/scripts/dev_start.sh
(qwecxz)
./docker/scripts/dev_into.sh
./apollo.sh build_opt_gpu
./scripts/bootstrap_lgsvl.sh
./scripts/bridge.sh
