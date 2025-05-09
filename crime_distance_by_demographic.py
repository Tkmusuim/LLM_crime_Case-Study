import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.environment.map import Map
import pickle
from collections import defaultdict
import matplotlib
from tqdm import tqdm
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''
本代码将分别绘制baseline和llm模型在不同人口学（种族、性别、种族×性别）特征上犯罪距离的差异，以柱状图形式表示。
更换不同baseline可以直接搜索替换关键字'baseline1/2/3'和'base1/2/3'
'''

# 初始化地图
def initialize_map():
    # 直接使用Map类初始化，参考map.py
    map_obj = Map('cache/map_data_Chicago.pkl', None)
    return map_obj

def calculate_crime_distances_by_demographic(records_dir, map_obj):
    """计算每个犯罪者两次犯罪之间的距离，按照种族和性别分类"""
    # 初始化按种族和性别分类的距离数据
    race_distances = defaultdict(list)
    gender_distances = defaultdict(list)
    race_gender_distances = defaultdict(list)
    
    # 获取所有符合条件的文件列表
    files = [f for f in os.listdir(records_dir) if f.endswith('_records.json')]
    
    # 使用tqdm添加进度条
    for filename in tqdm(files, desc=f"处理 {records_dir}", ncols=100):
        file_path = os.path.join(records_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"无法解析JSON文件: {file_path}")
                continue
        
        # 提取犯罪者的人口统计学特征
        profile = data.get('profile', {})
        race = profile.get('race', 'Unknown')
        gender = profile.get('gender', 'Unknown')
        agent_id = profile.get('agent_id', '')
        
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
        
        # 如果有距离记录，按照种族和性别分类记录
        if agent_distances:
            avg_distance = np.mean(agent_distances)
            race_distances[race].append(avg_distance)
            gender_distances[gender].append(avg_distance)
            race_gender_distances[f"{race}_{gender}"].append(avg_distance)
    
    return {
        'race': {race: np.mean(distances) for race, distances in race_distances.items()},
        'gender': {gender: np.mean(distances) for gender, distances in gender_distances.items()},
        'race_gender': {rg: np.mean(distances) for rg, distances in race_gender_distances.items()}
    }

def main():
    print("开始初始化地图...")
    # 初始化地图
    map_obj = initialize_map()
    print("地图初始化完成！")
    
    print("开始计算不同人口统计学特征的犯罪距离...")
    # 计算两个数据集的犯罪距离，按照种族和性别分类
    minus_score_demographics = calculate_crime_distances_by_demographic('results_minus_score_20250508/individual_records', map_obj)
    base1_demographics = calculate_crime_distances_by_demographic('results_base1/individual_records', map_obj)
    
    print("\n结果统计完成！开始绘制图表...")
    
    # 绘制种族分布柱状图
    plt.figure(figsize=(15, 10))
    
    # 1. 按种族分布的柱状图
    plt.subplot(2, 2, 1)
    races = sorted(set(list(minus_score_demographics['race'].keys()) + list(base1_demographics['race'].keys())))
    
    x = np.arange(len(races))
    width = 0.35
    
    llm_race_means = [minus_score_demographics['race'].get(race, 0) for race in races]
    base_race_means = [base1_demographics['race'].get(race, 0) for race in races]
    
    plt.bar(x - width/2, llm_race_means, width, label='LLM模型')
    plt.bar(x + width/2, base_race_means, width, label='baseline1')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(llm_race_means):
        plt.text(i - width/2, v + 100, f'{v:.0f}', ha='center', va='bottom')
    for i, v in enumerate(base_race_means):
        plt.text(i + width/2, v + 100, f'{v:.0f}', ha='center', va='bottom')
    
    plt.xlabel('种族')
    plt.ylabel('平均距离 (米)')
    plt.title('不同种族的犯罪距离对比')
    plt.xticks(x, races)
    plt.legend()
    
    # 2. 按性别分布的柱状图
    plt.subplot(2, 2, 2)
    genders = sorted(set(list(minus_score_demographics['gender'].keys()) + list(base1_demographics['gender'].keys())))
    
    x = np.arange(len(genders))
    
    llm_gender_means = [minus_score_demographics['gender'].get(gender, 0) for gender in genders]
    base_gender_means = [base1_demographics['gender'].get(gender, 0) for gender in genders]
    
    plt.bar(x - width/2, llm_gender_means, width, label='LLM模型')
    plt.bar(x + width/2, base_gender_means, width, label='baseline1')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(llm_gender_means):
        plt.text(i - width/2, v + 100, f'{v:.0f}', ha='center', va='bottom')
    for i, v in enumerate(base_gender_means):
        plt.text(i + width/2, v + 100, f'{v:.0f}', ha='center', va='bottom')
    
    plt.xlabel('性别')
    plt.ylabel('平均距离 (米)')
    plt.title('不同性别的犯罪距离对比')
    plt.xticks(x, genders)
    plt.legend()
    
    # 3. 按种族和性别组合的柱状图
    plt.subplot(2, 1, 2)
    race_genders = sorted(set(list(minus_score_demographics['race_gender'].keys()) + list(base1_demographics['race_gender'].keys())))
    
    x = np.arange(len(race_genders))
    
    llm_rg_means = [minus_score_demographics['race_gender'].get(rg, 0) for rg in race_genders]
    base_rg_means = [base1_demographics['race_gender'].get(rg, 0) for rg in race_genders]
    
    plt.bar(x - width/2, llm_rg_means, width, label='LLM模型')
    plt.bar(x + width/2, base_rg_means, width, label='baseline1')
    
    # 在柱状图上添加数值标签，只在值大于0时添加
    for i, v in enumerate(llm_rg_means):
        if v > 0:
            plt.text(i - width/2, v + 100, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(base_rg_means):
        if v > 0:
            plt.text(i + width/2, v + 100, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('种族_性别')
    plt.ylabel('平均距离 (米)')
    plt.title('不同种族和性别组合的犯罪距离对比')
    plt.xticks(x, race_genders, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('crime_distance_by_demographic.png', dpi=300)
    plt.close()
    
    print(f"图表已保存为 'crime_distance_by_demographic.png'")

    print("\n按种族统计的平均距离:")
    for race in races:
        llm_dist = minus_score_demographics['race'].get(race, 0)
        base_dist = base1_demographics['race'].get(race, 0)
        print(f"种族: {race}, LLM模型: {llm_dist:.2f}米, baseline1: {base_dist:.2f}米")
    
    print("\n按性别统计的平均距离:")
    for gender in genders:
        llm_dist = minus_score_demographics['gender'].get(gender, 0)
        base_dist = base1_demographics['gender'].get(gender, 0)
        print(f"性别: {gender}, LLM模型: {llm_dist:.2f}米, baseline1: {base_dist:.2f}米")

if __name__ == "__main__":
    main() 