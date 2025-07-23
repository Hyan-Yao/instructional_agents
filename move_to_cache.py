"""
Script: move_to_cache.py
Description:
    本脚本用于组织实验目录中的文件结构，自动将指定的 JSON 文件和 Markdown 结果报告文件
    移动至缓存目录 .cache 中，方便后续整理和备份。

Usage:
    python move_to_cache.py --exp <experiment_name>

Example:
    python move_to_cache.py --exp my_experiment

Functionality:
    1. 接收命令行参数 --exp, 指定实验名称
    2. 创建 ./exp/[exp_name]/.cache 目录（若已存在则跳过）
    3. 将 ./exp/[exp_name]/ 目录下（不包括 .cache 子目录）的所有
        - `.json` 文件；
        - 指定的 Markdown 文件：
            - result_assessment_planning.md
            - result_resource_assessment.md
            - result_target_audience.md
       移动到 .cache 目录中；
    4. 移动过程会在终端输出日志。
"""

import argparse
import os
import shutil

def move_files_to_cache(exp_name):
    base_dir = os.path.join('exp', exp_name)
    cache_dir = os.path.join(base_dir, '.cache')

    # 创建 .cache 目录
    os.makedirs(cache_dir, exist_ok=True)

    # 要移动的特定文件名
    special_files = {
        'result_assessment_planning.md',
        'result_resource_assessment.md',
        'result_target_audience.md'
    }

    # 遍历 base_dir 下的所有文件（不包含 .cache）
    for root, dirs, files in os.walk(base_dir):
        # 跳过 .cache 目录
        if os.path.abspath(root) == os.path.abspath(cache_dir):
            continue

        for file in files:
            file_path = os.path.join(root, file)

            # 如果是 .json 文件或特定的 .md 文件
            if file.endswith('.json') or file in special_files:
                target_path = os.path.join(cache_dir, file)
                print(f"Moving: {file_path} -> {target_path}")
                shutil.move(file_path, target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move files to cache folder under exp/exp_name/")
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    args = parser.parse_args()

    move_files_to_cache(args.exp)
