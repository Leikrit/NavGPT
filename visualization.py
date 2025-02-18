# import os
# import json

# def extract_trajectories_from_json(input_json_file):
#     """
#     从指定的JSON文件中提取所有字典中的"trajectory"项的第一个元素。
#     """
#     try:
#         with open(input_json_file, 'r', encoding='utf-8') as file:
#             data = json.load(file)
        
#         # 提取每个字典中的"trajectory"项的第一个元素
#         trajectories = [item['trajectory'][0] for item in data if 'trajectory' in item and item['trajectory']]
#         return trajectories
#     except Exception as e:
#         print(f"Error reading {input_json_file}: {e}")
#         return []

# def find_json_files_with_string(folder_path, search_string):
#     """
#     在指定文件夹中查找包含指定字符串的JSON文件。
#     """
#     matching_files = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     data = json.load(file)
#                     if search_string in json.dumps(data):
#                         matching_files.append(file_path)
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#     return matching_files

# def save_lists_to_file(output_file, trajectories, folder_path):
#     """
#     根据提取的字符串，查找对应的JSON文件并保存每个字符串对应的列表。
#     """
#     results = {}
#     for string in trajectories:
#         matching_files = find_json_files_with_string(folder_path, string[0])
#         for file_path in matching_files:
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     data = json.load(file)
#                     # 假设列表是文件中的一个键值对
#                     for item in data['nodes']:
#                         if isinstance(item, dict) and string[0] in item.get('trajectory', []):
#                             results[string[0]] = item['trajectory']
#                             break
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
    
#     # 将结果保存到文件
#     with open(output_file, 'w', encoding='utf-8') as file:
#         json.dump(results, file, indent=4)

# # 主程序
# if __name__ == "__main__":
#     input_json_file = "/home/lijinyi/NavGPT/datasets/R2R/exprs/ollama-qwen2-7b-test-2/preds/submit_R2R_val_unseen_instr.json"  # 输入的JSON文件路径
#     folder_path = "/home/lijinyi/NavGPT/datasets/R2R/habitat_mp3d_connectivity_graphs"  # 包含JSON文件的文件夹路径
#     output_file = "output_1.json"  # 输出文件路径

#     # 提取trajectories
#     trajectories = extract_trajectories_from_json(input_json_file)
#     if trajectories:
#         print(f"Extracted trajectories: {trajectories}")
#         # 保存每个字符串对应的列表
#         save_lists_to_file(output_file, trajectories, folder_path)
#         print(f"Results saved to {output_file}")
#     else:
#         print("No trajectories found in the input JSON file.")


import json
import os

def extract_trajectories_from_json(input_json_file):
    """
    从指定的JSON文件中提取所有字典中的"trajectory"项。
    """
    try:
        with open(input_json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 提取每个字典中的"trajectory"项
        trajectories = [item['trajectory'] for item in data if 'trajectory' in item]
        return trajectories
    except Exception as e:
        print(f"Error reading {input_json_file}: {e}")
        return []

def find_matching_json_file(folder_path, search_string):
    """
    在指定文件夹中查找包含指定字符串的JSON文件。
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # 检查该文件是否包含指定的节点字符串
                    if any(search_string in str(value) for value in data.values()):
                        return filename  # 返回文件名，而不是完整路径
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return None

def convert_trajectories_to_coordinates(trajectories, nodes_folder):
    """
    将每个trajectory中的节点标识符转换为对应的坐标，并查找对应的JSON文件。
    """
    results = []
    for idx, trajectory in enumerate(trajectories):
        trajectory_result = {"coordinates": []}
        for sublist in trajectory:
            if sublist:  # 跳过空列表
                node_id = sublist[0]
                # 查找对应的JSON文件
                matching_file = find_matching_json_file(nodes_folder, node_id)
                if matching_file:
                    file_path = os.path.join(nodes_folder, matching_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            nodes = json.load(file)
                            # 提取节点坐标
                            if node_id in nodes.get("nodes", {}):
                                trajectory_result["coordinates"].append(nodes["nodes"][node_id])
                            else:
                                print(f"Node ID {node_id} not found in {matching_file}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                else:
                    print(f"No matching JSON file found for node ID {node_id}")
        
        # 如果有有效坐标，则记录对应的文件名
        if trajectory_result["coordinates"]:
            trajectory_result["scene"] = matching_file if matching_file else "unknown.json"
        else:
            trajectory_result["scene"] = "unknown.json"
        
        results.append(trajectory_result)
    return results

def save_results_to_file(output_file, results):
    """
    将汇总结果保存到JSON文件中。
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)

# 主程序
if __name__ == "__main__":
    input_json_file = "/home/lijinyi/NavGPT/datasets/R2R/exprs/ollama-qwen2-7b-test-2/preds/submit_R2R_val_unseen_instr.json"  # 输入的JSON文件路径
    nodes_folder = "/home/lijinyi/NavGPT/datasets/R2R/habitat_mp3d_connectivity_graphs"  # 包含节点JSON文件的文件夹路径
    output_file = "output_1.json"  # 输出文件路径

    # 提取所有trajectories
    trajectories = extract_trajectories_from_json(input_json_file)
    if not trajectories:
        print("No trajectories found in the input JSON file.")
        exit()

    # 将trajectory中的节点标识符转换为坐标，并查找对应的JSON文件
    results = convert_trajectories_to_coordinates(trajectories, nodes_folder)

    # 保存结果
    save_results_to_file(output_file, results)
    print(f"Results saved to {output_file}")