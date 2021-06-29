'''
tune objectives
top down camera
'''


import os
import lgsvl
import time
import psutil
import atexit
import math
from .object_types import static_types, pedestrian_types, vehicle_types
from customized_utils import emptyobject
import numpy as np

accident_happen = False

# temporary, can be imported from customized_utils
def exit_handler():
    PROC_NAME = "mainboard"
    for proc in psutil.process_iter():
        # check whether the process to kill name matches
        if proc.name() == PROC_NAME:
            proc.kill()
##################################################


def norm_2d(loc_1, loc_2):
    return np.sqrt((loc_1.x - loc_2.x) ** 2 + (loc_1.z - loc_2.z) ** 2)

def on_waypoint(agent, index):
    print("Waypoint {} reached".format(index))



def initialize_simulator(map, sim_specific_arguments):
    SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", 8181))
    BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 9090))

    if not sim_specific_arguments.sim:
        sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
        sim_specific_arguments.sim = sim
    else:
        sim = sim_specific_arguments.sim

    if sim.current_scene == map:
        sim.reset()
    else:
        # seed make sure the weather and NPC behvaiors deterministic
        sim.load(map, seed=0)

    return sim, BRIDGE_HOST, BRIDGE_PORT


def initialize_dv_and_ego(sim, model_id, start, destination, BRIDGE_HOST, BRIDGE_PORT, events_path):

    global accident_happen
    accident_happen = False

    def on_collision(agent1, agent2, contact):
        global accident_happen
        accident_happen = True
        name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
        name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
        print("{} collided with {} at {}".format(name1, name2, contact))
        print('v_ego:', agent1.state.velocity)

        loc = agent1.transform.position
        if not agent2:
            other_agent_type = 'static'
        else:
            other_agent_type = agent2.name
        ego_speed = np.linalg.norm([agent1.state.velocity.x, agent1.state.velocity.y, agent1.state.velocity.z])
        # d_angle_norm = angle_from_center_view_fov(agent2, agent1)
        #
        # if d_angle_norm > 0:
        #     ego_speed = -1

        data_row = ['collision', ego_speed, other_agent_type, loc.x, loc.y]
        data_row = ','.join([str(data) for data in data_row])
        with open(events_path, 'a') as f_out:
            print('\n'*3, 'data_row', data_row, '\n'*3)
            f_out.write(data_row+'\n')


    state = lgsvl.AgentState()
    state.transform = start

    ego = sim.add_agent(model_id, lgsvl.AgentType.EGO, state)
    ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)
    ego.on_collision(on_collision)

    # Dreamview setup
    dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)
    dv.set_hd_map('Borregas Ave')
    dv.set_vehicle('Lincoln2017MKZ_LGSVL')
    modules = [
        'Localization',
        'Perception',
        'Transform',
        'Routing',
        'Prediction',
        'Planning',
        'Camera',
        # 'Traffic Light',
        'Control'
    ]

    start = lgsvl.Transform(position=ego.transform.position, rotation=ego.transform.rotation)

    print('start', start)
    print('destination', destination)
    dv.setup_apollo(destination.position.x, destination.position.z, modules, default_timeout=60)
    print('finish setup_apollo')

    return ego




def start_simulation(customized_data, arguments, sim_specific_arguments, launch_server, episode_max_time):


    events_path = os.path.join(arguments.deviations_folder, "events.txt")
    deviations_path = os.path.join(arguments.deviations_folder, 'deviations.txt')
    main_camera_folder = os.path.join(arguments.deviations_folder, 'main_camera_data')

    if not os.path.exists(main_camera_folder):
        os.mkdir(main_camera_folder)

    model_id = arguments.model_id
    map = arguments.route_info["town_name"]
    start, destination = arguments.route_info["location_list"]

    sim, BRIDGE_HOST, BRIDGE_PORT = initialize_simulator(map, sim_specific_arguments)

    try:
        sim.weather = lgsvl.WeatherState(rain=customized_data['rain'], fog=customized_data['fog'], wetness=customized_data['wetness'], cloudiness=customized_data['cloudiness'], damage=customized_data['damage'])
        sim.set_time_of_day(customized_data['hour'], fixed=True)
    except:
        import traceback
        traceback.print_exc()


    start = lgsvl.Transform(position=lgsvl.Vector(start[0], start[1], start[2]), rotation=lgsvl.Vector(start[3], start[4], start[5]))
    destination = lgsvl.Transform(position=lgsvl.Vector(destination[0], destination[1], destination[2]), rotation=lgsvl.Vector(destination[3], destination[4], destination[5]))

    ego = initialize_dv_and_ego(sim, model_id, start, destination, BRIDGE_HOST, BRIDGE_PORT, events_path)

    middle_point = lgsvl.Transform(position=(destination.position + start.position) * 0.5, rotation=start.rotation)


    other_agents = []
    for static in customized_data['static_list']:
        state = lgsvl.ObjectState()
        state.transform.position = lgsvl.Vector(static.x,0,static.y)
        state.transform.rotation = lgsvl.Vector(0,0,0)
        state.velocity = lgsvl.Vector(0,0,0)
        state.angular_velocity = lgsvl.Vector(0,0,0)


        static_object = sim.controllable_add(static_types[static.model], state)

    for ped in customized_data['pedestrians_list']:
        ped_position_offset = lgsvl.Vector(ped.x, 0, ped.y)
        ped_rotation_offset = lgsvl.Vector(0, 0, 0)
        ped_point = lgsvl.Transform(position=middle_point.position+ped_position_offset, rotation=middle_point.rotation+ped_rotation_offset)

        forward = lgsvl.utils.transform_to_forward(ped_point)

        wps = []
        for wp in ped.waypoints:
            loc = middle_point.position+lgsvl.Vector(wp.x, 0, wp.y)
            wps.append(lgsvl.WalkWaypoint(loc, 0, wp.trigger_distance))

        state = lgsvl.AgentState()
        state.transform = ped_point
        state.velocity = ped.speed * forward

        p = sim.add_agent(pedestrian_types[ped.model], lgsvl.AgentType.PEDESTRIAN, state)
        p.follow(wps, False)

        other_agents.append(p)

    for vehicle in customized_data['vehicles_list']:
        vehicle_position_offset = lgsvl.Vector(vehicle.x, 0, vehicle.y)
        vehicle_rotation_offset = lgsvl.Vector(0, 0, 0)
        vehicle_point = lgsvl.Transform(position=middle_point.position+vehicle_position_offset, rotation=middle_point.rotation+vehicle_rotation_offset)

        forward = lgsvl.utils.transform_to_forward(vehicle_point)

        wps = []
        for wp in vehicle.waypoints:
            loc = middle_point.position + lgsvl.Vector(wp.x, 0, wp.y)
            wps.append(lgsvl.DriveWaypoint(loc, vehicle.speed, lgsvl.Vector(0, 0, 0), 0, False, wp.trigger_distance))

        state = lgsvl.AgentState()
        state.transform = vehicle_point
        p = sim.add_agent(vehicle_types[vehicle.model], lgsvl.AgentType.NPC, state)
        p.follow(wps, False)

        other_agents.append(p)



    t0 = time.time()
    s0 = sim.current_time
    print()
    print("Total real time elapsed = {:5.3f}".format(0))
    print("Simulation time = {:5.1f}".format(s0))
    print("Simulation frames =", sim.current_frame)

    # let simulator initialize and settle a bit before starting
    # sim.run(time_limit=2)

    t1 = time.time()
    s1 = sim.current_time

    duration = 30 #episode_max_time
    step_time = 1
    step_rate = int(1.0 / step_time)
    steps = duration * step_rate


    cur_values = emptyobject(min_d=10000, d_angle_norm=1)


    for i in range(steps):

        sim.run(time_limit=step_time, time_scale=1)
        t2 = time.time()
        s2 = sim.current_time

        state = ego.state
        pos = state.position
        rot = state.rotation
        speed = state.speed * 3.6


        print("Sim time = {:5.2f}".format(s2 - s1) + "; Real time elapsed = {:5.3f}; ".format(t2 - t1), end='')
        print("Speed = {:4.1f}; Position = {:5.3f},{:5.3f},{:5.3f}; Rotation = {:5.3f},{:5.3f},{:5.3f}".format(speed, pos.x, pos.y, pos.z, rot.x, rot.y, rot.z))


        gather_info(ego, other_agents, cur_values, deviations_path)

        d_to_dest = norm_2d(ego.transform.position, destination.position)
        print('d_to_dest', d_to_dest)
        if d_to_dest < 5:
            print('ego car reachs destination successfully')
            break

        for sensor in ego.get_sensors():
            if sensor.name == "Main Camera":
                rel_path = "../2020_CARLA_challenge/"+main_camera_folder+"/"+"main_camera_"+str(i)+".png"
                sensor.save(rel_path, compression=9)
        if accident_happen:
            break
        time.sleep(0.2)
    sim.reset()


# def angle_from_center_view_fov(target, ego, fov=90):
#     target_location = target.transform.position
#     ego_location = ego.transform.position
#     ego_orientation = ego.transform.rotation.y
#
#     # hack: adjust to the front central camera's location
#     # this needs to be changed when the camera's location / fov change
#     dx = 1.3 * np.cos(np.deg2rad(ego_orientation - 90))
#
#
#     target_vector = np.array([target_location.x - ego_location.x + dx, target_location.z - ego_location.z])
#     print('target_location.x, ego_location.x, dx, target_location.z, ego_location.z', target_location.x, ego_location.x, dx, target_location.z, ego_location.z)
#     norm_target = np.linalg.norm(target_vector)
#
#     # modification: differ from current carla implementation
#     if norm_target < 0.001:
#         return 1
#
#     forward_vector = np.array(
#         [
#             math.cos(math.radians(ego_orientation)),
#             math.sin(math.radians(ego_orientation)),
#         ]
#     )
#
#     try:
#         d_angle = np.abs(
#             math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))
#         )
#     except:
#         print(
#             "\n" * 3,
#             "np.dot(forward_vector, target_vector)",
#             np.dot(forward_vector, target_vector),
#             norm_target,
#             "\n" * 3,
#         )
#         d_angle = 0
#     # d_angle_norm == 0 when target within fov
#     d_angle_norm = np.clip((d_angle - fov / 2) / (180 - fov / 2), 0, 1)
#
#     return d_angle_norm

def get_bbox(agent):

    # print('agent.bounding_box', agent.bounding_box)
    x_min = agent.bounding_box.min.x
    x_max = agent.bounding_box.max.x
    z_min = agent.bounding_box.min.z
    z_max = agent.bounding_box.max.z
    bbox = [
        agent.transform.position+lgsvl.Vector(x_min, 0, z_min),
        agent.transform.position+lgsvl.Vector(x_min, 0, z_max),
        agent.transform.position+lgsvl.Vector(x_max, 0, z_min),
        agent.transform.position+lgsvl.Vector(x_max, 0, z_max)
    ]

    return bbox

def gather_info(ego, other_agents, cur_values, deviations_path):
    print('gather_info', 'deviations_path', deviations_path)
    ego_bbox = get_bbox(ego)
    # TBD: only using the front two vertices
    # ego_front_bbox = ego_bbox[:2]

    min_d = 9999
    d_angle_norm = 0.99
    for i, other_agent in enumerate(other_agents):

        # d_angle_norm_i = angle_from_center_view_fov(other_agent, ego, fov=90)
        # d_angle_norm = np.min([d_angle_norm, d_angle_norm_i])
        # if d_angle_norm_i == 0:
        d_angle_norm = 0
        other_bbox = get_bbox(other_agent)
        for other_b in other_bbox:
            for ego_b in ego_bbox:
                # print('other_bbox, ego_bbox', other_bbox, ego_bbox)
                d = norm_2d(other_b, ego_b)
                min_d = np.min([min_d, d])

    print('min_d', min_d)
    print('d_angle_norm', d_angle_norm)
    if min_d < cur_values.min_d:
        cur_values.min_d = min_d
        with open(deviations_path, 'a') as f_out:
            f_out.write('min_d,'+str(cur_values.min_d)+'\n')

    if d_angle_norm < cur_values.d_angle_norm:
        cur_values.d_angle_norm = d_angle_norm
        with open(deviations_path, 'a') as f_out:
            f_out.write('d_angle_norm,'+str(cur_values.d_angle_norm)+'\n')


    # TBD: out-of-road violation related data




if __name__ == '__main__':
    atexit.register(exit_handler)
    map = "BorregasAve"
    config = [4, 4, 2, 3, 10, 50]
    run_svl_simulation(map, config)
