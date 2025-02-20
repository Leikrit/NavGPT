import math
import os
import random
import time
import uuid

import cv2
import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# data_path = "./data"
# print(f"data_path = {data_path}")
# # @markdown Optionally configure the save path for video output:
# output_directory = "examples"
# output_path = os.path.join('/home/zhandijia/DockerData/zhandijia-root/ETPNav-lijinyi', output_directory)
# if not os.path.exists(output_path):
#     os.mkdir(output_path)

# def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
#     from habitat_sim.utils.common import d3_40_colors_rgb

#     rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

#     arr = [rgb_img]
#     titles = ["rgb"]
#     if semantic_obs.size != 0:
#         semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
#         semantic_img.putpalette(d3_40_colors_rgb.flatten())
#         semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
#         semantic_img = semantic_img.convert("RGBA")
#         arr.append(semantic_img)
#         titles.append("semantic")

#     if depth_obs.size != 0:
#         depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
#         arr.append(depth_img)
#         titles.append("depth")

#     plt.figure(figsize=(12, 8))
#     for i, data in enumerate(arr):
#         ax = plt.subplot(1, 3, i + 1)
#         ax.axis("off")
#         ax.set_title(titles[i])
#         plt.imshow(data)
#     plt.show(block=False)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--no-display", dest="display", action="store_false")
#     parser.add_argument("--no-make-video", dest="make_video", action="store_false")
#     parser.set_defaults(show_video=True, make_video=True)
#     args, _ = parser.parse_known_args()
#     show_video = args.display
#     display = args.display
#     do_make_video = args.make_video
# else:
#     show_video = False
#     do_make_video = False
#     display = False

# # import the maps module alone for topdown mapping
# if display:
#     from habitat.utils.visualizations import maps

# test_scene = "./data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"

# sim_settings = {
#     "scene": test_scene,  # Scene path
#     "default_agent": 0,  # Index of the default agent
#     "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
#     "width": 256,  # Spatial resolution of the observations
#     "height": 256,
# }

# def make_simple_cfg(settings):
#     # simulator backend
#     sim_cfg = habitat_sim.SimulatorConfiguration()
#     sim_cfg.scene_id = settings["scene"]

#     # agent
#     agent_cfg = habitat_sim.agent.AgentConfiguration()

#     # In the 1st example, we attach only one sensor,
#     # a RGB visual sensor, to the agent
#     rgb_sensor_spec = habitat_sim.SensorSpec()
#     rgb_sensor_spec.uuid = "color_sensor"
#     rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
#     rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
#     rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

#     agent_cfg.sensor_specifications = [rgb_sensor_spec]

#     return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# cfg = make_simple_cfg(sim_settings)

# try:  # Needed to handle out of order cell run in Colab
#     sim.close()
# except NameError:
#     pass
# sim = habitat_sim.Simulator(cfg)

# agent = sim.initialize_agent(sim_settings["default_agent"])

# # Set agent state
# agent_state = habitat_sim.AgentState()
# agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
# agent.set_state(agent_state)

# # Get agent state
# agent_state = agent.get_state()
# print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
# print("Discrete action space: ", action_names)


# def navigateAndSee(action=""):
#     if action in action_names:
#         Obeservation = sim.get_state()
#         observations = sim.step(action)
#         print("action: ", action)
#         bgr_data = cv2.cvtColor(observations['color_sensor'][:, :, :3], cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
#         cv2.imwrite(os.path.join(output_path, f'imgs/{int(time.time() * 1000)}.png'), bgr_data)
#         if display:
#             display_sample(observations["color_sensor"])


# action = "turn_right"
# navigateAndSee(action)

# action = "turn_right"
# navigateAndSee(action)

# action = "move_forward"
# navigateAndSee(action)

# action = "turn_left"
# navigateAndSee(action)

# # convert 3d points to 2d topdown coordinates
# def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
#     points_topdown = []
#     bounds = pathfinder.get_bounds()
#     for point in points:
#         # convert 3D x,z to topdown x,y
#         px = (point[0] - bounds[0][0]) / meters_per_pixel
#         py = (point[2] - bounds[0][2]) / meters_per_pixel
#         points_topdown.append(np.array([px, py]))
#     return points_topdown


# # display a topdown map with matplotlib
# def display_map(topdown_map, key_points=None):
#     plt.figure(figsize=(12, 8))
#     ax = plt.subplot(1, 1, 1)
#     ax.axis("off")
#     plt.imshow(topdown_map)
#     # plot points on map
#     if key_points is not None:
#         for point in key_points:
#             plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
#     plt.show(block=False)


# # @markdown ###Configure Example Parameters:
# # @markdown Configure the map resolution:
# meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# # @markdown ---
# # @markdown Customize the map slice height (global y coordinate):
# custom_height = False  # @param {type:"boolean"}
# height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
# # @markdown If not using custom height, default to scene lower limit.
# # @markdown (Cell output provides scene height range from bounding box for reference.)

# print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
# if not custom_height:
#     # get bounding box minumum elevation for automatic height
#     height = sim.pathfinder.get_bounds()[0][1]

# if not sim.pathfinder.is_loaded:
#     print("Pathfinder not initialized, aborting.")
# else:
#     # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
#     # This map is a 2D boolean array
#     sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

#     if display:
#         # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-api/blob/master/habitat/utils/visualizations/maps.py)
#         hablab_topdown_map = maps.get_topdown_map(
#             sim.pathfinder, height, meters_per_pixel=meters_per_pixel
#         )
#         recolor_map = np.array(
#             [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
#         )
#         hablab_topdown_map = recolor_map[hablab_topdown_map]
#         print("Displaying the raw map from get_topdown_view:")
#         display_map(sim_topdown_map)
#         print("Displaying the map from the Habitat-Lab maps module:")
#         display_map(hablab_topdown_map)

#         # easily save a map to file:
#         map_filename = os.path.join(output_path, "top_down_map.png")
#         imageio.imsave(map_filename, hablab_topdown_map)

data_path = ""

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def extract_coordinates_by_scene(output_json_file):
    """
    从output.json文件中按scene提取坐标，并去掉scene字段值中的.json后缀。
    """
    try:
        with open(output_json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 初始化一个字典来存储每个scene的坐标
        coordinates_by_scene = {}
        
        # 遍历每个trajectory的结果
        for item in data:
            scene = item.get("scene", "unknown")  # 获取scene字段
            # 去掉scene字段值中的.json后缀
            scene = scene.replace('.json', '')  
            coordinates = item.get("coordinates", [])  # 获取coordinates字段
            
            # 如果scene已经存在，追加坐标；否则创建新的键
            if scene in coordinates_by_scene:
                coordinates_by_scene[scene].extend(coordinates)
            else:
                coordinates_by_scene[scene] = coordinates
        
        return coordinates_by_scene
    except Exception as e:
        print(f"Error reading {output_json_file}: {e}")
        return {}

def save_coordinates_by_scene_to_file(coordinates_by_scene, output_file):
    """
    将按scene提取的坐标保存到一个新的JSON文件中。
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(coordinates_by_scene, file, indent=4)
    print(f"Coordinates by scene saved to {output_file}")

def get_panorama(sim, agent_state, num_views=36, view_angle=360):
    """
    获取全景图
    :param sim: HabitatSim 模拟器实例
    :param agent_state: Agent 的初始状态（位置和旋转）
    :param num_views: 捕获的视角数量（默认36个视角）
    :param view_angle: 捕获的总角度（默认360度）
    :return: 拼接后的全景图
    """
    # 初始化全景图列表
    panorama_images = []

    # 计算每个视角的旋转角度
    step_angle = math.radians(view_angle / num_views)

    for i in range(num_views):
        # 计算当前视角的旋转角度
        current_angle = i * step_angle

        # 更新 Agent 的旋转状态
        agent_state.rotation.y = current_angle
        sim.set_agent_state(agent_state)

        # 获取当前视角的观测
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]

        # 将图像从 Habitat 的格式转换为 OpenCV 的格式
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 添加到全景图列表
        panorama_images.append(rgb)

    # 拼接全景图
    panorama = np.hstack(panorama_images)
    return panorama

# 主程序
if __name__ == "__main__":
    output_json_file = "output.json"  # 之前的输出文件
    coordinates_output_file = f"coordinates_by_scene_from_{output_json_file.replace('.json', '')}.json"  # 保存按scene分组的坐标文件
    output_path = "panorama.jpg"  # 全景图指定保存路径和文件名

    # 按scene提取坐标
    coordinates_by_scene = extract_coordinates_by_scene(output_json_file)
    if coordinates_by_scene:
        print(f"Extracted coordinates for {len(coordinates_by_scene)} scenes.")
        # 保存按scene分组的坐标到文件
        save_coordinates_by_scene_to_file(coordinates_by_scene, coordinates_output_file)

        for scene, coordinates in coordinates_by_scene.items():
            print(f"================Processing scene {scene}...===================")
            test_scene = os.path.join(
                data_path, f"scene_datasets/mp3d_example/{scene}/{scene}.glb"
            )
            sim_settings = {
                "scene": test_scene,  # Scene path
                "default_agent": 0,  # Index of the default agent
                "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
                "width": 256,  # Spatial resolution of the observations
                "height": 256,
            } 
            cfg = make_simple_cfg(sim_settings)
            sim = habitat_sim.Simulator(cfg)



            path_points = coordinates
            # # @markdown - Success, geodesic path length, and 3D points can be queried.
            # print("found_path : " + str(found_path))
            # print("geodesic_distance : " + str(geodesic_distance))
            # print("path_points : " + str(path_points))

            # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
            meters_per_pixel = 0.025
            height = sim.scene_aabb.y().min
            display = True
            if display:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
                recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
                top_down_map = recolor_map[top_down_map]
                grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                # convert world trajectory points to maps module grid points
                trajectory = [
                    maps.to_grid(
                        path_point[2],
                        path_point[0],
                        grid_dimensions,
                        pathfinder=sim.pathfinder,
                    )
                    for path_point in path_points
                ]
                grid_tangent = mn.Vector2(
                    trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                )
                path_initial_tangent = grid_tangent / grid_tangent.length()
                initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                # draw the agent and trajectory on the map
                maps.draw_path(top_down_map, trajectory)
                maps.draw_agent(
                    top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                )
                print("\nDisplay the map with agent and path overlay:")
                display_map(top_down_map)

                # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
                display_path_agent_renders = True  # @param{type:"boolean"}
                if display_path_agent_renders:
                    print("Rendering observations at path points:")
                    tangent = path_points[1] - path_points[0]
                    agent_state = habitat_sim.AgentState()
                    for ix, point in enumerate(path_points):
                        if ix < len(path_points) - 1:
                            tangent = path_points[ix + 1] - point
                            agent_state.position = point
                            tangent_orientation_matrix = mn.Matrix4.look_at(
                                point, point + tangent, np.array([0, 1.0, 0])
                            )
                            tangent_orientation_q = mn.Quaternion.from_matrix(
                                tangent_orientation_matrix.rotation()
                            )
                            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                            agent.set_state(agent_state)

                            # 获取全景图
                            panorama = get_panorama(sim, agent_state)

                            if display_panorama:
                                # 显示全景图
                                cv2.imshow("Panorama", panorama)

                                # 保存全景图到文件
                                scene_panorama = os.path.join(
                                    f"{scene}_", output_path
                                )
                                cv2.imwrite(scene_panorama, panorama)
                                print(f"Panorama saved to {scene_panorama}")
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                            observations = sim.get_sensor_observations()
                            rgb = observations["color_sensor"]

                            if display:
                                display_sample(rgb)
    
    else:
        print("No coordinates found in the output JSON file.")


