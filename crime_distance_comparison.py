import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.environment.map import Map
import pickle
from collections import defaultdict
import matplotlib
from tqdm import tqdm  # 添加tqdm进度条
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''
本代码用于统计和对比LLM模型和baseline模型的Agent平均两次犯罪距离对比
更换不同baseline请搜索替换'base1/2/3'
'''

# 初始化地图
def initialize_map():
    # 直接使用Map类初始化，参考map.py
    map_obj = Map('cache/map_data_Chicago.pkl', None)
    return map_obj

def calculate_crime_distances(records_dir, map_obj):
    """计算每个犯罪者两次犯罪之间的距离"""
    all_distances = []
    agent_avg_distances = {}
    
    # 获取所有符合条件的文件列表
    files = [f for f in os.listdir(records_dir) if f.endswith('_records.json')]

    for filename in tqdm(files, desc=f"处理 {records_dir}", ncols=100):
        file_path = os.path.join(records_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"无法解析JSON文件: {file_path}")
                continue
        
        # 提取agent_id
        agent_id = data.get('profile', {}).get('agent_id', '')
        
        # 提取犯罪记录
        records = data.get('records', {}).get('records', [])
        
        # 按步骤排序
        records.sort(key=lambda x: x.get('step', 0))
        
        # 如果记录少于2条，则跳过
        if len(records) < 2:
            continue
        
        # 计算每两次犯罪之间的距离
        agent_distances = []
        for i in range(1, len(records)):
            prev_location = records[i-1]['location']
            curr_location = records[i]['location']
            
            # 使用Map对象计算两个CBG之间的距离
            distance = map_obj.calculate_distance(prev_location, curr_location)
            
            if distance != float('inf'):
                agent_distances.append(distance)
        
        # 如果有距离记录，计算该犯罪者的平均距离
        if agent_distances:
            avg_distance = np.mean(agent_distances)
            agent_avg_distances[agent_id] = avg_distance
            all_distances.extend(agent_distances)
    
    return all_distances, agent_avg_distances

def main():
    print("开始初始化地图...")
    # 初始化地图
    map_obj = initialize_map()
    print("地图初始化完成！")
    
    print("开始计算犯罪距离...")
    # 计算两个数据集的犯罪距离
    minus_score_distances, minus_score_avg = calculate_crime_distances('results_minus_score_20250508/individual_records', map_obj)
    base3_distances, base3_avg = calculate_crime_distances('results_base3/individual_records', map_obj)
    
    # 统计分析
    minus_score_mean = np.mean(minus_score_distances) if minus_score_distances else 0
    base3_mean = np.mean(base3_distances) if base3_distances else 0
    
    minus_score_median = np.median(minus_score_distances) if minus_score_distances else 0
    base3_median = np.median(base3_distances) if base3_distances else 0
    
    print(f"\n结果统计:")
    print(f"results_minus_score_20250508: 均值={minus_score_mean:.2f}米, 中位数={minus_score_median:.2f}米, 样本数={len(minus_score_distances)}")
    print(f"results_base3: 均值={base3_mean:.2f}米, 中位数={base3_median:.2f}米, 样本数={len(base3_distances)}")
    
    print("开始绘制图表...")
    # 绘制柱状图
    plt.figure(figsize=(14, 8))
    
    # 距离分布柱状图
    plt.subplot(2, 1, 1)
    bins = np.linspace(0, max(max(minus_score_distances or [0]), max(base3_distances or [0])), 30)
    plt.hist(minus_score_distances, bins=bins, alpha=0.7, label='LLM模型 (results_minus_score_20250508)')
    plt.hist(base3_distances, bins=bins, alpha=0.7, label='基线模型 (results_base3)')
    plt.xlabel('距离 (米)')
    plt.ylabel('频率')
    plt.legend()
    plt.title('两次犯罪之间的距离分布对比')
    
    # 平均距离柱状图
    plt.subplot(2, 1, 2)
    means = [minus_score_mean, base3_mean]
    medians = [minus_score_median, base3_median]
    labels = ['LLM模型\n(results_minus_score_20250508)', '基线模型\n(results_base3)']
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='均值')
    plt.bar(x + width/2, medians, width, label='中位数')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(means):
        plt.text(i - width/2, v + 100, f'{v:.0f}', ha='center', va='bottom')
    for i, v in enumerate(medians):
        plt.text(i + width/2, v + 100, f'{v:.0f}', ha='center', va='bottom')
    
    plt.xlabel('数据集')
    plt.ylabel('距离 (米)')
    plt.title('两次犯罪之间的平均距离和中位距离')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('crime_distance_comparison.png', dpi=300)
    plt.close()
    
    print(f"图表已保存为 'crime_distance_comparison.png'")

if __name__ == "__main__":
    main() 