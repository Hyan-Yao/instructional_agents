import os
import json
import numpy as np

def read_json_file(file_path):
    """读取json文件并返回数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_summary_average(json_data, section):
    """计算给定section的average_score"""
    if section in json_data:
        return json_data[section]['summary']['average_score']
    return None

def calculate_average_and_std(data):
    """计算数据的平均值和标准差"""
    return np.mean(data), np.std(data)

def analyze_folders(folders):
    """分析给定文件夹列表中的所有evaluation_scores.json文件"""
    section_averages = {
        'learning_objectives': [],
        'syllabus': [],
        'assessment': [],
        'slide_content': [],
        'slide_scripts': [],
        'overall_summary': []
    }
    
    # 遍历文件夹列表
    for folder in folders:
        json_file_path = os.path.join(folder, 'evaluation_results', 'evaluation_scores.json')
        
        if os.path.exists(json_file_path):
            data = read_json_file(json_file_path)
            
            for section in section_averages:
                avg_score = calculate_summary_average(data, section)
                if avg_score is not None:
                    section_averages[section].append(avg_score)
    
    # 计算每个子项的平均值和标准差
    results = {}
    for section, scores in section_averages.items():
        if scores:
            avg, std = calculate_average_and_std(scores)
            results[section] = {
                'average': avg,
                'std_dev': std,
                'scores': scores
            }
    
    return results

# 示例用法
folders = ['A_data_mining_Sent', 'B_foundations_of_machine_learning_Sent', 'C_data_processing_at_scale_Sent', 'D_Introduction_to_Artificial_Intelligence_Wanpeng_07022025_Sent', 'E_topics_in_reinforcement_learning_Sent']  # 请替换为实际的文件夹路径列表
folders = ['gpt-4o_A_data_mining_V1_Wanpeng_0718', 'gpt-4o_B_Foundations_of_machine_learning_V1_Wanpeng_0718', 'gpt-4o_C_data_processing_at_scale_V1_Wanpeng_0718', 'gpt-4o_D_Introduction_to_Artificial_Intelligence_V1_Wanpeng_0718', 'gpt-4o_E_topics_in_reinforcement_learning_V1_Wanpeng_0718']
folders = ['o1-preview_A_data_mining_V1_Wanpeng_0718', 'o1-preview_B_Foundations_of_machine_learning_V1_Wanpeng_0718', 'o1-preview_C_data_processing_at_scale_V1_Wanpeng_0718', 'o1-preview_D_Introduction_to_Artificial_Intelligence_V1_Wanpeng_0718', 'o1-preview_E_topics_in_reinforcement_learning_V1_Wanpeng_0718']
results = analyze_folders(folders)

# 输出结果
for section, result in results.items():
    print(f"{section}: Average = {result['average']}, Std Dev = {result['std_dev']}, scores = {result['scores']}")
