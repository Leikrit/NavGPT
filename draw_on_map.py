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
from habitat.utils.visualizations import maps
from habitat_sim.utils import viz_utils as vut

import json
import re
from scipy.spatial.transform import Rotation as R

# =====================Configurations==========================
data_path = "/home/zhandijia/DockerData/zhandijia-root/ETPNav/data/scene_datasets/mp3d/"
output_json_file = "output_1.json"  # 之前的输出文件
coordinates_output_file = f"coordinates_by_scene_from_{output_json_file.replace('.json', '')}.json"  # 保存按scene分组的坐标文件
output_path = "panorama.jpg"  # 全景图指定保存路径和文件名
save_path = "imgs3" # 保存可视化图片文件夹

def extract_scene_name(scene_key):
    """
    使用正则表达式提取scene名称并去掉标识符（如_1, _2等）。
    """
    match = re.match(r"^(.*)_([0-9]+)$", scene_key)
    if match:
        return match.group(1)  # 返回去掉标识符的scene名称
    return scene_key  # 如果没有标识符，直接返回原始名称

def display_sample(rgb_obs, save_img_path, index, semantic_obs=np.array([]), depth_obs=np.array([])):
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
        ax = plt.subplot(1, 2, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # 检查图像是否为灰度图像
        print(i)
        if i == 1:  # 灰度图像
            plt.imshow(data, cmap='gray')  # 使用灰度颜色映射
        else:
            plt.imshow(data)  # 彩色图像
    # plt.show(block=False)
    plt.savefig(f"{save_img_path}_{index}_original.jpg")

# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

# display a topdown map with matplotlib
def display_map(topdown_map, save_map_path, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # plt.show(block=False)
    plt.savefig(f"{save_map_path}_map.png")

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
    rgb_sensor_spec = habitat_sim.SensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# def extract_coordinates_by_scene(output_json_file):
#     """
#     从output.json文件中按scene提取坐标，并将每个坐标点转换为ndarray。
#     """
#     try:
#         with open(output_json_file, 'r', encoding='utf-8') as file:
#             data = json.load(file)
#
#         # 初始化一个字典来存储每个scene的坐标
#         coordinates_by_scene = {}
#
#         # 遍历每个trajectory的结果
#         for item in data:
#             scene = item.get("scene", "unknown")  # 获取scene字段
#             # 去掉scene字段值中的.json后缀
#             scene = scene.replace('.json', '')
#             coordinates = item.get("coordinates", [])  # 获取coordinates字段
#
#             # 将每个坐标点转换为ndarray
#             coordinates = [np.array(coord) for coord in coordinates]
#
#             # 如果scene已经存在，追加坐标；否则创建新的键
#             if scene in coordinates_by_scene:
#                 coordinates_by_scene[scene].extend(coordinates)
#             else:
#                 coordinates_by_scene[scene] = coordinates
#
#         return coordinates_by_scene
#     except Exception as e:
#         print(f"Error reading {output_json_file}: {e}")
#         return {}

def extract_coordinates_by_scene(output_json_file):
    """
    从output.json文件中按scene提取坐标，并将每个坐标点转换为ndarray。
    每个scene的轨迹将分开存储，并为每个scene添加唯一标识符。
    """
    try:
        with open(output_json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 初始化一个字典来存储每个scene的坐标
        coordinates_by_scene = {}
        scene_counter = {}  # 用于记录每个scene的计数

        # 遍历每个trajectory的结果
        for item in data:
            scene = item.get("scene", "unknown")  # 获取scene字段
            # 去掉scene字段值中的.json后缀
            scene = scene.replace('.json', '')

            # 为scene添加唯一标识符
            if scene not in scene_counter:
                scene_counter[scene] = 0
            scene_counter[scene] += 1
            unique_scene = f"{scene}_{scene_counter[scene]}"

            coordinates = item.get("coordinates", [])  # 获取coordinates字段
            coordinates = [np.array(coord) for coord in coordinates]  # 转换为ndarray

            coordinates_by_scene[unique_scene] = coordinates

        return coordinates_by_scene
    except Exception as e:
        print(f"Error reading {output_json_file}: {e}")
        return {}

# def save_coordinates_by_scene_to_file(coordinates_by_scene, output_file):
#     """
#     将按scene提取的坐标保存到一个新的JSON文件中。
#     """
#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(coordinates_by_scene, file, indent=4)
#     print(f"Coordinates by scene saved to {output_file}")

# def save_coordinates_by_scene_to_file(coordinates_by_scene, output_file):
#     """
#     将按scene提取的坐标保存到一个新的JSON文件中。
#     """
#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(coordinates_by_scene, file, indent=4, default=lambda x: x.tolist())
#     print(f"Coordinates by scene saved to {output_file}")


def save_coordinates_by_scene_to_file(coordinates_by_scene, output_file):
    """
    将按scene提取的坐标保存到一个新的JSON文件中。
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(coordinates_by_scene, file, indent=4, default=lambda x: x.tolist())
    print(f"Coordinates by scene saved to {output_file}")

def angle_to_quaternion(angle_degrees):
    """
    将角度转换为四元数
    :param angle_degrees: 绕Y轴旋转的角度（单位：度）
    :return: 四元数表示的旋转
    """
    angle_radians = np.deg2rad(angle_degrees)
    rotation = R.from_euler('y', angle_radians)
    quaternion = rotation.as_quat()  # 返回四元数 [x, y, z, w]
    return np.quaternion(quaternion[3], quaternion[0], quaternion[1], quaternion[2])

def get_panorama(sim, agent_state, save_path, index, num_views=12, view_angle=360, depth=False):
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
    panorama_depth = []

    # 计算每个视角的旋转角度
    step_angle = view_angle / num_views
    print(step_angle)

    for i in range(num_views):
        # 计算当前视角的旋转角度
        current_angle = i * step_angle
        rotation_quaternion = angle_to_quaternion(current_angle)
        # 更新 Agent 的旋转状态
        # agent_state.rotation.y = current_angle
        agent_state.rotation = rotation_quaternion
        agent.set_state(agent_state)

        # 获取当前视角的观测
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        if depth:
            depth_obs = observations["depth_sensor"]
            depth_obs = np.array(depth_obs, dtype=np.uint8)
            panorama_depth.append(depth_obs)

        # 将图像从 Habitat 的格式转换为 NumPy 数组
        rgb = np.array(rgb, dtype=np.uint8)

        # 添加到全景图列表
        panorama_images.append(rgb)

    # # 将 numpy 数组转换为 PIL 图像
    # pil_images = [Image.fromarray(img) for img in panorama_images]
    # # 获取图像的宽度和高度
    # widths, heights = zip(*(i.size for i in pil_images))
    # # 计算总宽度和最大高度
    # total_width = sum(widths)
    # max_height = max(heights)
    #
    # # 创建一个新的空白图像
    # new_im = Image.new('RGB', (total_width, max_height))
    #
    # # 拼接图像
    # x_offset = 0
    # for img in pil_images:
    #     new_im.paste(img, (x_offset, 0))
    #     x_offset += img.size[0]
    #
    # # 保存全景图
    # new_im.save('panorama.jpg')
    # print("全景图拼接成功，保存为 panorama.jpg")
    rgb_imgs = []
    for x in panorama_images:
        xi = Image.fromarray(x, mode="RGBA")
        xii = xi.convert("RGB")
        rgb_array = np.array(xii)
        rgb_imgs.append(rgb_array)
    if depth:
        depth_imgs = []
        for x in panorama_depth:
            xi = Image.fromarray(x, mode="L")
            depth_array = np.array(xi)
            depth_imgs.append(depth_array)

    # rgb_images = panorama_images[:, :, :3]
    # bgr_images = cv2.cvtColor(rgb_images, cv2.COLOR_RGB2BGR)
    # 创建拼接器
    stitcher = cv2.Stitcher_create()

    # 拼接图像
    try:
        status, panorama_full = stitcher.stitch(rgb_imgs)
        if status == cv2.Stitcher_OK:
            # 保存全景图
            cv2.imwrite(f'{save_path}_{index}_generated_panorama.jpg', panorama_full)
            print(f"全景图拼接成功，保存为{save_path}_{index}_generated_panorama.jpg")
        else:
            print("拼接失败，错误代码：", status)
    except Exception as e:
        print("Failed to stitch RGB images:", e)

    # if depth:
    #     try:
    #         status, panorama_full = stitcher.stitch(depth_imgs)
    #         if status == cv2.Stitcher_OK:
    #             # 保存全景图
    #             cv2.imwrite(f'{save_path}_{index}_generated_depth_panorama.jpg', panorama_full)
    #             print(f"全景图拼接成功，保存为{save_path}_{index}_generated_depth_panorama.jpg")
    #         else:
    #             print("拼接失败，错误代码：", status)
    #     except Exception as e:
    #         print("Failed to stitch depth images:", e)

    if not depth:
        # 使用 matplotlib 拼接全景图
        fig, ax = plt.subplots(figsize=(80, 5))  # 调整大小以适应全景图
        ax.imshow(np.hstack(panorama_images))
        ax.axis('off')  # 关闭坐标轴
        plt.tight_layout()
        return fig
    else:
        fig, ax = plt.subplots(figsize=(80, 10))
        ax = plt.subplot(2, 1, 1)
        ax.axis("off")
        ax.imshow(np.hstack(panorama_images))
        ax = plt.subplot(2, 1, 2)
        ax.axis("off")
        ax.imshow(np.hstack(panorama_depth), cmap="gray")
        plt.tight_layout()
        return fig


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        # "semantic_sensor": {
        #     "sensor_type": habitat_sim.SensorType.SEMANTIC,
        #     "resolution": [settings["height"], settings["width"]],
        #     "position": [0.0, settings["sensor_height"], 0.0],
        # },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# 主程序
if __name__ == "__main__":

    # 按scene提取坐标
    coordinates_by_scene = extract_coordinates_by_scene(output_json_file)
    if coordinates_by_scene:
        print(f"Extracted coordinates for {len(coordinates_by_scene)} scenes.")
        # 保存按scene分组的坐标到文件
        save_coordinates_by_scene_to_file(coordinates_by_scene, coordinates_output_file)
        for scenekey, coordinates in coordinates_by_scene.items():
            scene = extract_scene_name(scenekey)
            if scene == "unknown":
                continue
            print(f"================Processing scene {scene}...===================")
            current_path = os.getcwd()
            save_scene = os.path.join(current_path, save_path, scenekey)
            if os.path.exists(save_scene):
                pass
            else:
                try:
                    os.makedirs(save_scene)
                    print(f"已创建路径：{save_scene}")
                except Exception as e:
                    print(f"创建路径{save_scene}时出错：{e}")

            test_scene = os.path.join(
                data_path, f"{scene}/{scene}.glb"
            )
            rgb_sensor = True  # @param {type:"boolean"}
            depth_sensor = True # @param {type:"boolean"}
            semantic_sensor = False  # @param {type:"boolean"}

            sim_settings = {
                "width": 256,  # Spatial resolution of the observations
                "height": 256,
                "scene": test_scene,  # Scene path
                "default_agent": 0,
                "sensor_height": coordinates[0][1] + 1.5,  # Height of sensors in meters
                "color_sensor": rgb_sensor,  # RGB sensor
                "depth_sensor": depth_sensor,  # Depth sensor
                "seed": 1,  # used in the random navigation
                "enable_physics": False,  # kinematics only
            }


            # cfg = make_simple_cfg(sim_settings)
            try:
                # cfg = make_simple_cfg(sim_settings)
                cfg = make_cfg(sim_settings)
            except ValueError as e:
                print(f"Could not find scene {scene}: {e}")
                continue
            sim = habitat_sim.Simulator(cfg)
            # initialize an agent
            agent = sim.initialize_agent(sim_settings["default_agent"])


            path_points = coordinates
            # # @markdown - Success, geodesic path length, and 3D points can be queried.
            # print("found_path : " + str(found_path))
            # print("geodesic_distance : " + str(geodesic_distance))
            # print("path_points : " + str(path_points))

            # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
            meters_per_pixel = 0.025
            # scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            # height = scene_bb.y().min
            # print(f"Height_bb: {height}")
            # print(f"Height_coo: {coordinates[0][1]}")
            height = coordinates[0][1]
            base_map_path = os.path.join(save_scene, f"{scenekey}")

            display = True
            if display:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height
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
                display_map(top_down_map, base_map_path)

                # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
                display_path_agent_renders = True  # @param{type:"boolean"}
                if display_path_agent_renders:
                    print("Rendering observations at path points:")
                    tangent = path_points[1] - path_points[0]
                    agent_state = habitat_sim.AgentState()
                    temp_tangent = None
                    for ix, point in enumerate(path_points):
                        base_path = os.path.join(save_scene, f"{scenekey}_{ix}")
                        if ix < len(path_points) - 1:
                            if (point == path_points[ix+1]).all():
                                continue
                            tangent = path_points[ix + 1] - point
                            agent_state.position = point
                            tangent_orientation_matrix = mn.Matrix4.look_at(
                                point, point + tangent, np.array([0, 1.0, 0])
                            )
                            tangent_orientation_q = mn.Quaternion.from_matrix(
                                tangent_orientation_matrix.rotation()
                            )
                            temp_tangent = tangent_orientation_q
                            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                            agent.set_state(agent_state)
                            observations = sim.get_sensor_observations()
                            rgb = observations["color_sensor"]
                            depth = observations["depth_sensor"]
                            if display:
                                display_sample(rgb, base_path, ix, depth_obs=depth)
                                display_panorama = True
                                if display_panorama:
                                    # 获取并显示全景图
                                    panorama = get_panorama(sim, agent_state, base_path, ix, depth=True)
                                    scene_panorama = os.path.join(save_scene, f"{scenekey}_{ix}_" + output_path)
                                    panorama.savefig(scene_panorama, bbox_inches='tight', pad_inches=0)
                        else:
                            agent_state.position = point
                            agent_state.rotation = utils.quat_from_magnum(temp_tangent)
                            agent.set_state(agent_state)
                            observations = sim.get_sensor_observations()
                            rgb = observations["color_sensor"]
                            depth = observations["depth_sensor"]
                            if display:
                                display_sample(rgb, base_path, ix, depth_obs=depth)
                                display_panorama = True
                                if display_panorama:
                                    # 获取并显示全景图
                                    panorama = get_panorama(sim, agent_state, base_path, ix, depth=True)
                                    scene_panorama = os.path.join(save_scene, f"{scenekey}_{ix}_" + output_path)
                                    panorama.savefig(scene_panorama, bbox_inches='tight', pad_inches=0)

                sim.close()
    else:
        print("No coordinates found in the output JSON file.")

