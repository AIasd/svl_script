import os
import pickle
import numpy as np
import lgsvl
from .object_params import Pedestrian, Vehicle, Static, Waypoint
from customized_utils import make_hierarchical_dir, emptyobject, check_bug, classify_bug_type
from .scene_configs import customized_bounds_and_distributions, customized_routes
from .simulation_utils import start_simulation
import shutil


def convert_x_to_customized_data(
    x,
    fuzzing_content,
    port
):

    waypoints_num_limit = fuzzing_content.search_space_info.waypoints_num_limit
    num_of_static_max = fuzzing_content.search_space_info.num_of_static_max
    num_of_pedestrians_max = fuzzing_content.search_space_info.num_of_pedestrians_max
    num_of_vehicles_max = fuzzing_content.search_space_info.num_of_vehicles_max

    customized_center_transforms = fuzzing_content.customized_center_transforms
    parameters_min_bounds = fuzzing_content.parameters_min_bounds
    parameters_max_bounds = fuzzing_content.parameters_max_bounds

    # parameters
    # global

    num_of_static = int(x[0])
    num_of_pedestrians = int(x[1])
    num_of_vehicles = int(x[2])
    damage = int(x[3])
    rain = x[4]
    fog = x[5]
    wetness = x[6]
    cloudiness = x[7]
    hour = x[8]

    ind = 9

    # static
    static_list = []
    for i in range(num_of_static_max):
        if i < num_of_static:
            static_i = Static(
                model=int(x[ind]),
                x=x[ind+1],
                y=x[ind+2],
            )
            static_list.append(static_i)
        ind += 3

    # pedestrians
    pedestrians_list = []
    for i in range(num_of_pedestrians_max):
        if i < num_of_pedestrians:
            pedestrian_type_i = int(x[ind])
            pedestrian_x_i = x[ind+1]
            pedestrian_y_i = x[ind+2]
            pedestrian_speed_i = x[ind+3]
            ind += 4

            pedestrian_waypoints_i = []
            for _ in range(waypoints_num_limit):
                pedestrian_waypoints_i.append(Waypoint(x[ind], x[ind+1], x[ind+2]))
                ind += 3

            pedestrian_i = Pedestrian(
                model=pedestrian_type_i,
                x=pedestrian_x_i,
                y=pedestrian_y_i,
                speed=pedestrian_speed_i,
                waypoints=pedestrian_waypoints_i,
            )

            pedestrians_list.append(pedestrian_i)

        else:
            ind += 4 + waypoints_num_limit * 3

    # vehicles
    vehicles_list = []
    for i in range(num_of_vehicles_max):
        if i < num_of_vehicles:
            vehicle_type_i = int(x[ind])
            vehicle_x_i = x[ind+1]
            vehicle_y_i = x[ind+2]
            vehicle_speed_i = x[ind+3]
            ind += 4

            vehicle_waypoints_i = []
            for _ in range(waypoints_num_limit):
                vehicle_waypoints_i.append(Waypoint(x[ind], x[ind+1], x[ind+2]))
                ind += 3

            vehicle_i = Vehicle(
                model=vehicle_type_i,
                x=vehicle_x_i,
                y=vehicle_y_i,
                speed=vehicle_speed_i,
                waypoints=vehicle_waypoints_i,
            )

            vehicles_list.append(vehicle_i)

        else:
            ind += 4 + waypoints_num_limit * 3




    customized_data = {
        # "num_of_static": num_of_static,
        # "num_of_pedestrians": num_of_pedestrians,
        # "num_of_vehicles": num_of_vehicles,
        "damage": damage,
        "rain": rain,
        "fog": fog,
        "wetness": wetness,
        "cloudiness": cloudiness,
        "hour": hour,
        "static_list": static_list,
        "pedestrians_list": pedestrians_list,
        "vehicles_list": vehicles_list,
        "customized_center_transforms": customized_center_transforms,
        # "parameters_min_bounds": parameters_min_bounds,
        # "parameters_max_bounds": parameters_max_bounds,
    }

    return customized_data


def estimate_objectives(save_path, default_objectives=np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.]), verbose=True):

    events_path = os.path.join(save_path, "events.txt")
    deviations_path = os.path.join(save_path, "deviations.txt")

    # set thresholds to avoid too large influence
    ego_linear_speed = 0
    min_d = 20
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0
    d_angle_norm = 1

    ego_linear_speed_max = 7
    dev_dist_max = 7

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0
    is_collision = 0
    if os.path.exists(deviations_path):
        with open(deviations_path, "r") as f_in:
            for line in f_in:
                type, d = line.split(",")
                d = float(d)
                if type == "min_d":
                    min_d = np.min([min_d, d])
                elif type == "offroad_d":
                    offroad_d = np.min([offroad_d, d])
                elif type == "wronglane_d":
                    wronglane_d = np.min([wronglane_d, d])
                elif type == "dev_dist":
                    dev_dist = np.max([dev_dist, d])
                elif type == "d_angle_norm":
                    d_angle_norm = np.min([d_angle_norm, d])

    x = None
    y = None
    object_type = None

    if not os.path.exists(events_path):
        route_completion = True
    else:
        route_completion = False
        with open(events_path, 'r') as f_in:
            tokens = f_in.read().split('\n')[0].split(',')
            _, ego_linear_speed, object_type, x, y = tokens
            ego_linear_speed, x, y = float(ego_linear_speed), float(x), float(y)
        if ego_linear_speed > 0.1:
            is_collision = 1


    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, ego_linear_speed_max])
    dev_dist = np.min([dev_dist, dev_dist_max])

    return (
        [
            ego_linear_speed,
            min_d,
            d_angle_norm,
            offroad_d,
            wronglane_d,
            dev_dist,
            is_collision,
            is_offroad,
            is_wrong_lane,
            is_run_red_light,
        ],
        (x, y),
        object_type,
        route_completion,
    )


def run_svl_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):
    '''
    objectives needs to be consistent with the specified objectives



    not using:
    launch_server,
    port

    '''
    print('\n'*3, 'x:\n', x, '\n'*3)
    customized_data = convert_x_to_customized_data(x, fuzzing_content, port)
    parent_folder = fuzzing_arguments.parent_folder
    episode_max_time = fuzzing_arguments.episode_max_time
    mean_objectives_across_generations_path = fuzzing_arguments.mean_objectives_across_generations_path
    ego_car_model = fuzzing_arguments.ego_car_model


    # 5.0: 47b529db-0593-4908-b3e7-4b24a32a0f70
    # 6.0: c354b519-ccf0-4c1c-b3cc-645ed5751bb5
    # 6.0(modular testing): 2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921
    # 6.0(no telephoto camera and clock sensor): 4622f73a-250e-4633-9a3d-901ede6b9551
    # 6.0(no clock sensor): f68151d1-604c-438e-a1a5-aa96d5581f4b
    # 6.0(with signal sensor): 9272dd1a-793a-45b2-bff4-3a160b506d75
    # 6.0(modular testing, birdview): b20c0d8a-f310-46b2-a639-6ce6be4f2b14
    if ego_car_model == 'apollo_6_with_signal':
        model_id = '9272dd1a-793a-45b2-bff4-3a160b506d75'
    elif ego_car_model == 'apollo_6_modular':
        model_id = '2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921'


    route_info = sim_specific_arguments.route_info
    deviations_folder = os.path.join(parent_folder, "current_run_data")
    if os.path.exists(deviations_folder):
        shutil.rmtree(deviations_folder)
    os.mkdir(deviations_folder)

    arguments = emptyobject(deviations_folder=deviations_folder, model_id=model_id, route_info=route_info, record_every_n_step=fuzzing_arguments.record_every_n_step)



    start_simulation(customized_data, arguments, sim_specific_arguments, launch_server, episode_max_time)
    objectives, loc, object_type, route_completion = estimate_objectives(deviations_folder)



    if parent_folder:
        is_bug = check_bug(objectives)
        if is_bug:
            bug_type, bug_str = classify_bug_type(objectives, object_type)
        else:
            bug_type, bug_str = None, None
        if is_bug:
            with open(mean_objectives_across_generations_path, 'a') as f_out:
                f_out.write(str(counter)+','+bug_str+'\n')

        bug_folder = make_hierarchical_dir([parent_folder, 'bugs'])
        non_bug_folder = make_hierarchical_dir([parent_folder, 'non_bugs'])
        if is_bug:
            cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
        else:
            cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])



    xl = [pair[1] for pair in fuzzing_content.parameters_min_bounds.items()]
    xu = [pair[1] for pair in fuzzing_content.parameters_max_bounds.items()]

    import copy
    sim_specific_arguments_copy = copy.copy(sim_specific_arguments)
    sim_specific_arguments_copy.sim = None
    run_info = {
        # for analysis
        'x': x,
        'objectives': objectives,
        'labels': fuzzing_content.labels,

        'is_bug': is_bug,
        'bug_type': bug_type,

        'xl': np.array(xl),
        'xu': np.array(xu),
        'mask': fuzzing_content.mask,

        # for rerun
        'fuzzing_content': fuzzing_content,
        'fuzzing_arguments': fuzzing_arguments,
        'sim_specific_arguments': sim_specific_arguments_copy,
        'dt_arguments': dt_arguments,

        # helpful info
        'route_completion': route_completion,

        # for correction
        # 'all_final_generated_transforms': all_final_generated_transforms,
    }

    from distutils.dir_util import copy_tree
    copy_tree(deviations_folder, cur_folder)

    with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
        pickle.dump(run_info, f_out)




    return objectives, run_info


def initialize_svl_specific(fuzzing_arguments):
    route_info = customized_routes[fuzzing_arguments.route_type]
    sim_specific_arguments = emptyobject(route_info=route_info, sim=None)
    return sim_specific_arguments
