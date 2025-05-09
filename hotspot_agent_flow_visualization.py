import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union
import pandas as pd
import os
import traceback
import re
from collections import Counter, defaultdict
from matplotlib.lines import Line2D
import random

'''
本代码用于绘制LLM模型预测对，但baseline模型预测错的热点中，不同模型的Agent移动流向图
替换不同baseline请更换'base1/2/3'
'''

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

CBG_META_PATH = 'cache/cbg_meta_img_sampled_summary.json'
CBG_CRIME_PATH = 'cache/cbg_crime.pkl'
CHICAGO_MAP_DATA_PATH = 'cache/map_data_Chicago.pkl'
RESULTS_COT_PATH = 'results_minus_score_20250508/crime_records.json'
RESULTS_BASE_PATH = 'results_base2/crime_records.json'
COT_INDIVIDUAL_DIR = 'results_minus_score_20250508/individual_records'
BASE_INDIVIDUAL_DIR = 'results_base2/individual_records'
OUTPUT_DIR = 'hotspot_agent_flow_results'  # 输出目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 安全地打开JSON文件
def safe_load_json(file_path):
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"文件 {file_path} 不存在或为空")
            return None
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的JSON格式")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

# 读取犯罪数据并聚合到CBG
def preprocess_results(results_file):
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            crime_records = json.load(f)

        crime_results = {}
        for crime in crime_records:
            location = crime.get('location')
            if location:
                if location in crime_results:
                    crime_results[location] += 1
                else:
                    crime_results[location] = 1

        if not crime_results:
            return {}
            
        # 归一化
        total = sum(crime_results.values())
        crime_results = {k: v/total for k, v in crime_results.items()}
        # 排序
        crime_results = {k: v for k, v in sorted(crime_results.items(), key=lambda item: item[1], reverse=True)}
        return crime_results
    except Exception as e:
        print(f"处理犯罪数据时出错: {str(e)}")
        return {}

# 计算热点并找出LLM预测对但baseline预测错的CBG
def find_llm_correct_baseline_wrong(results_cot, results_base, ground_truth, threshold):
    # 检查是否有足够的数据
    if not ground_truth or not results_cot or not results_base:
        print("警告：没有足够的数据进行比较")
        return set()
    
    # 计算热点数量
    hotspots_num = max(1, int(len(ground_truth) * threshold))
    print(f"基于阈值 {threshold} 选取前 {hotspots_num} 个热点")
    
    # 获取每个数据集的热点
    ground_truth = {k: v for k, v in sorted(ground_truth.items(), key=lambda item: item[1], reverse=True)}
    gt_hotspots = set(list(ground_truth.keys())[:hotspots_num])
    
    results_cot = {k: v for k, v in sorted(results_cot.items(), key=lambda item: item[1], reverse=True)}
    cot_hotspots = set(list(results_cot.keys())[:hotspots_num])
    
    results_base = {k: v for k, v in sorted(results_base.items(), key=lambda item: item[1], reverse=True)}
    base_hotspots = set(list(results_base.keys())[:hotspots_num])
    
    # 找出LLM预测对但baseline预测错的CBG
    llm_correct_base_wrong = gt_hotspots & cot_hotspots - base_hotspots
    
    # 打印统计信息
    print(f"Ground Truth热点CBG数量: {len(gt_hotspots)}")
    print(f"LLM热点CBG数量: {len(cot_hotspots)}")
    print(f"Baseline热点CBG数量: {len(base_hotspots)}")
    print(f"LLM预测对但Baseline预测错的CBG数量: {len(llm_correct_base_wrong)}")
    
    return llm_correct_base_wrong

# 获取CBG描述
def get_cbg_description(cbg_id, cbg_meta_file):
    try:
        if not os.path.exists(cbg_meta_file):
            print(f"CBG元数据文件 {cbg_meta_file} 不存在")
            return "无法获取描述信息 - 文件不存在"
            
        if os.path.getsize(cbg_meta_file) == 0:
            print(f"CBG元数据文件 {cbg_meta_file} 为空")
            return "无法获取描述信息 - 文件为空"
            
        with open(cbg_meta_file, 'r', encoding='utf-8') as f:
            cbg_meta = json.load(f)
        
        if cbg_id in cbg_meta and 'meta' in cbg_meta[cbg_id]:
            # 获取meta字段的内容
            meta_content = cbg_meta[cbg_id]['meta']
            
            # 检查meta_content类型并适当处理
            if isinstance(meta_content, str):
                return meta_content
            elif isinstance(meta_content, dict) or isinstance(meta_content, list):
                # 如果是字典或列表，转为字符串
                return json.dumps(meta_content, ensure_ascii=False)
            else:
                # 其他类型直接转为字符串
                return str(meta_content)
        return f"无可用描述 (CBG ID: {cbg_id})"
    except json.JSONDecodeError:
        print(f"CBG元数据文件 {cbg_meta_file} 不是有效的JSON格式")
        return "无法获取描述信息 - 文件格式错误"
    except Exception as e:
        print(f"获取CBG描述时出错: {str(e)}")
        return f"无法获取描述信息 - {str(e)}"

# 收集访问热点的agent及其居住地
def collect_agent_flows(cbg_id, directory):
    """
    收集访问指定CBG的agent及其居住地
    返回: 包含agent_id和居住地的列表
    """
    agent_flows = []
    
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return agent_flows
    
    # 跳过不存在的文件
    files_to_skip = []
    if directory == COT_INDIVIDUAL_DIR:
        files_to_skip = ["C0_records.json"]
    
    # 获取所有json文件
    file_list = [f for f in os.listdir(directory) if f.endswith('_records.json') and f not in files_to_skip]
    
    for file in file_list:
        file_path = os.path.join(directory, file)
        data = safe_load_json(file_path)
        
        if data is None:
            continue
            
        try:
            # 检查agent的profile，获取居住地
            agent_id = None
            residence = None
            
            # 尝试从不同路径获取信息
            if 'profile' in data and isinstance(data['profile'], dict):
                agent_id = data['profile'].get('agent_id')
                residence = data['profile'].get('residence')
            elif 'agent_id' in data:
                agent_id = data['agent_id']
                if 'residence' in data:
                    residence = data['residence']
            
            # 如果仍未找到必要信息，跳过该文件
            if not agent_id or not residence:
                print(f"警告: 无法从 {file} 中获取agent_id或residence信息")
                continue
                
            # 检查agent是否访问过这个CBG
            has_visited = False
            
            # 尝试不同的数据结构来查找records
            records_to_check = []
            
            if 'records' in data:
                if isinstance(data['records'], list):
                    records_to_check = data['records']
                elif isinstance(data['records'], dict) and 'records' in data['records']:
                    records_to_check = data['records']['records']
            
            # 检查是否访问过目标CBG
            for record in records_to_check:
                if isinstance(record, dict) and record.get('location') == cbg_id:
                    has_visited = True
                    break
            
            if has_visited:
                agent_flows.append({
                    'agent_id': agent_id,
                    'residence': residence
                })
                print(f"发现agent {agent_id} 从居住地 {residence} 访问了CBG {cbg_id}")
                
        except Exception as e:
            print(f"处理文件 {file} 中的agent流数据时出错: {str(e)}")
    
    print(f"CBG {cbg_id} 被 {len(agent_flows)} 个不同的agents访问")
    return agent_flows

# 获取CBG的几何中心点
def get_cbg_centroid(cbg_id, chicago_data):
    if 'cbgs' in chicago_data and isinstance(chicago_data['cbgs'], dict):
        cbg_info = chicago_data['cbgs'].get(cbg_id)
        if cbg_info:
            try:
                geom = None
                # 尝试从不同字段获取几何信息
                if 'shapely_lnglat' in cbg_info and cbg_info['shapely_lnglat'] is not None:
                    geom = cbg_info['shapely_lnglat']
                elif 'geometry' in cbg_info:
                    if isinstance(cbg_info['geometry'], pd.Series) and len(cbg_info['geometry']) > 0:
                        geom_obj = cbg_info['geometry'].iloc[0]
                        if hasattr(geom_obj, 'geoms'):
                            geom = list(geom_obj.geoms)[0]
                        else:
                            geom = geom_obj
                    elif isinstance(cbg_info['geometry'], dict):
                        geom = shape(cbg_info['geometry'])
                    elif isinstance(cbg_info['geometry'], str):
                        try:
                            geom_data = json.loads(cbg_info['geometry'])
                            geom = shape(geom_data)
                        except json.JSONDecodeError:
                            return None
                
                if geom is not None and not geom.is_empty:
                    return geom.centroid
            except Exception as e:
                print(f"获取CBG {cbg_id}中心点时出错: {str(e)}")
    return None

# 在地图上可视化热点和agent流向
def visualize_agent_flows(cbg_id, llm_agent_flows, base_agent_flows, chicago_data, cbg_description, output_prefix):
    try:
        # 创建一个大图
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # 提取目标CBG的几何形状和芝加哥全部CBG
        all_geometries = []
        cbg_geometry = None
        all_centroids = {}
        residence_cbgs = set()  # 收集所有agent居住地的CBG ID
        
        # 收集所有agent的居住地CBG ID
        for flow in llm_agent_flows + base_agent_flows:
            if 'residence' in flow and flow['residence']:
                residence_cbgs.add(str(flow['residence']).strip())
        
        # 打印所有CBG ID 前10个(调试用)
        #print("Chicago数据中的前10个CBG ID:")
        cbg_ids = list(chicago_data.get('cbgs', {}).keys())

        #print(cbg_ids[:10] if len(cbg_ids) > 10 else cbg_ids)
        #print("第一个居住地ID:", next(iter(residence_cbgs)) if residence_cbgs else "无")
        
        # 收集所有CBG的几何形状
        if 'cbgs' in chicago_data and isinstance(chicago_data['cbgs'], dict):
            for id, cbg_info in chicago_data['cbgs'].items():
                id_str = str(id).strip()
                
                try:
                    geom = None
                    if 'shapely_lnglat' in cbg_info and cbg_info['shapely_lnglat'] is not None:
                        geom = cbg_info['shapely_lnglat']
                    elif 'geometry' in cbg_info:
                        if isinstance(cbg_info['geometry'], pd.Series) and len(cbg_info['geometry']) > 0:
                            geom_obj = cbg_info['geometry'].iloc[0]
                            if hasattr(geom_obj, 'geoms'):
                                geom = list(geom_obj.geoms)[0]
                            else:
                                geom = geom_obj
                        elif isinstance(cbg_info['geometry'], dict):
                            geom = shape(cbg_info['geometry'])
                        elif isinstance(cbg_info['geometry'], str):
                            try:
                                geom_data = json.loads(cbg_info['geometry'])
                                geom = shape(geom_data)
                            except json.JSONDecodeError:
                                continue
                    
                    if geom is not None and not geom.is_empty:
                        all_geometries.append(geom)
                        all_centroids[id_str] = geom.centroid
                        
                        # 标记目标CBG
                        if id_str == str(cbg_id).strip():
                            cbg_geometry = geom
                except Exception as e:
                    continue
        
        # 如果找不到目标CBG，无法绘制流向图
        if not cbg_geometry:
            print(f"警告：找不到CBG {cbg_id}的几何形状，无法绘制流向图")
            return

        print(f"居住地CBG数量: {len(residence_cbgs)}")
        print(f"找到的居住地中心点数量: {sum(1 for res in residence_cbgs if res in all_centroids)}")
        print(f"所有中心点数量: {len(all_centroids)}")
        
        # 检查哪些居住地没有找到中心点
        missing_centroids = [res for res in residence_cbgs if res not in all_centroids]
        if missing_centroids:
            print(f"找不到以下居住地的中心点: {missing_centroids[:5]}...")
        
        # 创建GeoDataFrame
        gdf_data = []
        
        # 添加所有CBG作为背景
        for geom in all_geometries:
            if geom != cbg_geometry:  # 排除目标CBG
                gdf_data.append({
                    'CBG': 'background',
                    'type': 0,  # 背景
                    'geometry': geom
                })
        
        # 添加目标CBG
        gdf_data.append({
            'CBG': cbg_id,
            'type': 1,  # 目标CBG
            'geometry': cbg_geometry
        })
        
        # 突出显示居住地CBG
        residence_geom_data = []
        for res_id in residence_cbgs:
            # 在chicago_data中查找居住地
            res_found = False
            for chicago_id, cbg_info in chicago_data.get('cbgs', {}).items():
                chicago_id_str = str(chicago_id).strip()
                if chicago_id_str == res_id:
                    res_found = True
                    if res_id != str(cbg_id).strip():  # 不是目标CBG
                        try:
                            geom = None
                            if 'shapely_lnglat' in cbg_info and cbg_info['shapely_lnglat'] is not None:
                                geom = cbg_info['shapely_lnglat']
                            elif 'geometry' in cbg_info:
                                if isinstance(cbg_info['geometry'], dict):
                                    geom = shape(cbg_info['geometry'])
                                elif isinstance(cbg_info['geometry'], str):
                                    try:
                                        geom_data = json.loads(cbg_info['geometry'])
                                        geom = shape(geom_data)
                                    except:
                                        pass
                            
                            if geom is not None and not geom.is_empty:
                                residence_geom_data.append({
                                    'CBG': res_id,
                                    'type': 2,  # 居住地CBG
                                    'geometry': geom
                                })
                        except Exception as e:
                            print(f"处理居住地CBG {res_id}时出错: {str(e)}")
                    break
            
            if not res_found:
                print(f"在chicago_data中找不到居住地CBG {res_id}")
        
        # 添加居住地CBG
        if residence_geom_data:
            gdf_data.extend(residence_geom_data)
            print(f"添加了 {len(residence_geom_data)} 个居住地CBG到地图")
        
        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(gdf_data, geometry='geometry')
        gdf.set_crs("EPSG:4326", inplace=True)
        
        # 绘制地图
        # 先绘制所有CBG作为背景
        background = gdf[gdf['type'] == 0]
        if not background.empty:
            background.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5, zorder=1)
        
        # 绘制居住地CBG
        residences = gdf[gdf['type'] == 2]
        if not residences.empty:
            residences.plot(ax=ax, color='#E0F2F1', edgecolor='darkgray', linewidth=0.8, zorder=2)
        
        # 绘制目标CBG（热点）
        target = gdf[gdf['type'] == 1]
        if not target.empty:
            target.plot(ax=ax, color='#004D40', edgecolor='black', linewidth=1.5, zorder=4)
        
        # 获取目标CBG的中心点
        target_centroid = all_centroids.get(str(cbg_id).strip())
        if not target_centroid:
            print(f"警告: 无法获取目标CBG {cbg_id}的中心点")
            return
        
        # 创建收集流线的列表
        llm_lines = []
        base_lines = []
        
        # 生成居住地到目标CBG的流向线
        # 1. 绘制LLM的agent流向
        for flow in llm_agent_flows:
            residence = str(flow.get('residence')).strip()
            if residence and residence in all_centroids:
                origin_centroid = all_centroids[residence]
                
                # 创建一条从居住地到目标CBG的线
                if origin_centroid and target_centroid:
                    # 创建直线（两点之间）
                    line = LineString([
                        (origin_centroid.x, origin_centroid.y),
                        (target_centroid.x, target_centroid.y)
                    ])
                    llm_lines.append(line)
                    print(f"创建LLM流向线 从 {residence} 到 {cbg_id}")
            else:
                print(f"LLM agent {flow.get('agent_id')} 的居住地 {residence} 找不到中心点")
        
        # 2. 绘制Baseline的agent流向
        for flow in base_agent_flows:
            residence = str(flow.get('residence')).strip()
            if residence and residence in all_centroids:
                origin_centroid = all_centroids[residence]
                
                # 创建一条从居住地到目标CBG的线
                if origin_centroid and target_centroid:
                    # 创建直线（两点之间）
                    line = LineString([
                        (origin_centroid.x, origin_centroid.y),
                        (target_centroid.x, target_centroid.y)
                    ])
                    base_lines.append(line)
                    print(f"创建Baseline流向线 从 {residence} 到 {cbg_id}")
            else:
                print(f"Baseline agent {flow.get('agent_id')} 的居住地 {residence} 找不到中心点")
        
        print(f"LLM流向线数量: {len(llm_lines)}")
        print(f"Baseline流向线数量: {len(base_lines)}")
        
        # 绘制流线 - 增加线条宽度和可见度
        if llm_lines:
            # 创建包含所有LLM流线的GeoDataFrame
            llm_lines_gdf = gpd.GeoDataFrame(geometry=llm_lines)
            llm_lines_gdf.set_crs("EPSG:4326", inplace=True)
            llm_lines_gdf.plot(ax=ax, color='#007bff', linewidth=2.0, alpha=0.8, zorder=3)
            print("绘制LLM流向线成功")
            
        if base_lines:
            # 创建包含所有Baseline流线的GeoDataFrame
            base_lines_gdf = gpd.GeoDataFrame(geometry=base_lines)
            base_lines_gdf.set_crs("EPSG:4326", inplace=True)
            base_lines_gdf.plot(ax=ax, color='#ff4757', linewidth=2.0, alpha=0.8, zorder=3)
            print("绘制Baseline流向线成功")
        
        # 为居住地CBG添加标签(仅选择前5个最常见的)
        residence_counts = Counter([flow['residence'] for flow in llm_agent_flows + base_agent_flows])
        top_residences = [str(res).strip() for res, _ in residence_counts.most_common(5)]
        
        # 为热点添加标签，但简化显示
        if target_centroid:
            '''
            ax.text(target_centroid.x, target_centroid.y, f"热点", 
                   fontsize=12, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
                   '''
        
        # 添加突出显示的箭头标记流向
        if llm_lines or base_lines:
            # 在流向线的中点添加箭头
            for i, line in enumerate(llm_lines):
                if i < 20:  # 限制箭头数量，避免过度拥挤
                    try:
                        mid_point = line.interpolate(0.5, normalized=True)
                        plt.arrow(mid_point.x, mid_point.y, 
                                 (target_centroid.x - mid_point.x) * 0.05, 
                                 (target_centroid.y - mid_point.y) * 0.05,
                                 head_width=0.005, head_length=0.008, 
                                 fc='#007bff', ec='#007bff', zorder=5)
                    except Exception as e:
                        print(f"绘制LLM流向线箭头时出错: {str(e)}")
            
            for i, line in enumerate(base_lines):
                if i < 20:  # 限制箭头数量，避免过度拥挤
                    try:
                        mid_point = line.interpolate(0.5, normalized=True)
                        plt.arrow(mid_point.x, mid_point.y, 
                                 (target_centroid.x - mid_point.x) * 0.05, 
                                 (target_centroid.y - mid_point.y) * 0.05,
                                 head_width=0.005, head_length=0.008, 
                                 fc='#ff4757', ec='#ff4757', zorder=5)
                    except Exception as e:
                        print(f"绘制Baseline流向线箭头时出错: {str(e)}")
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], color='#007bff', lw=2, label='LLM模型agent流向'),
            Line2D([0], [0], color='#ff4757', lw=2, label='Baseline模型agent流向'),
            Line2D([0], [0], marker='s', color='#004D40', lw=0, label='热点CBG',
                  markersize=10, markerfacecolor='#004D40'),
            Line2D([0], [0], marker='s', color='#E0F2F1', lw=0, label='Agent居住地CBG',
                  markersize=10, markerfacecolor='#E0F2F1', markeredgecolor='darkgray')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 设置标题
        ax.set_title(f'CBG - LLM预测对但Baseline预测错的热点的Agent流向')
        ax.set_axis_off()

        if cbg_description:
            if isinstance(cbg_description, dict):
                desc_str = str(cbg_description)
            else:
                desc_str = str(cbg_description)
            
            # 限制长度
            if len(desc_str) > 100:
                desc_str = desc_str[:100] + "..."
                
            plt.figtext(0.5, 0.01, f"热点描述: {desc_str}", 
                     ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # 统计信息
        info_text = (
            f"LLM模型: {len(llm_agent_flows)} 个agents\n"
            f"Baseline模型: {len(base_agent_flows)} 个agents"
        )
        plt.figtext(0.02, 0.02, info_text, 
                 fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":5})
        
        # 保存图像
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部的注脚留出空间
        output_file = f"{output_prefix}_agent_flow.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存Agent流向图到 {output_file}")
        
    except Exception as e:
        print(f"为CBG {cbg_id}生成Agent流向图时出错: {str(e)}")
        traceback.print_exc()

def main():
    try:
        print("开始加载数据...")
        
        # 加载CBG数据
        try:
            with open(CBG_META_PATH, 'r') as f:
                cbg_meta = json.load(f)
            print(f"成功加载CBG元数据，包含 {len(cbg_meta)} 个CBG")
            cbg_to_eval = [key for key in cbg_meta.keys() 
                          if 'meta' in cbg_meta[key] and len(cbg_meta[key]['meta']) >= 4]
            print(f"符合评估条件的CBG数量: {len(cbg_to_eval)}")
        except Exception as e:
            print(f"加载CBG元数据时出错: {str(e)}")
            cbg_meta = {}
            cbg_to_eval = []
        
        # 加载ground truth数据
        try:
            with open(CBG_CRIME_PATH, 'rb') as f:
                cbg_crime = pickle.load(f)
            print(f"成功加载真实犯罪数据，包含 {len(cbg_crime)} 个CBG")
            
            # 过滤和归一化ground truth数据
            ground_truth = {k: v for k, v in cbg_crime.items() if k in cbg_to_eval}
            if ground_truth:
                ground_truth = {k: v/sum(ground_truth.values()) for k, v in ground_truth.items()}
                print(f"过滤后的ground truth数据包含 {len(ground_truth)} 个CBG")
            else:
                print("警告：过滤后没有ground truth数据")
        except Exception as e:
            print(f"加载真实犯罪数据时出错: {str(e)}")
            ground_truth = {}
        
        # 加载LLM和Baseline预测结果
        try:
            cot_results = preprocess_results(RESULTS_COT_PATH)
            print(f"成功加载LLM预测结果，包含 {len(cot_results)} 个CBG")
        except Exception as e:
            print(f"加载LLM预测结果时出错: {str(e)}")
            cot_results = {}
            
        try:
            base_results = preprocess_results(RESULTS_BASE_PATH)
            print(f"成功加载Baseline预测结果，包含 {len(base_results)} 个CBG")
        except Exception as e:
            print(f"加载Baseline预测结果时出错: {str(e)}")
            base_results = {}
        
        # 加载芝加哥地图数据
        try:
            with open(CHICAGO_MAP_DATA_PATH, 'rb') as f:
                chicago_data = pickle.load(f)
            print(f"成功加载芝加哥地图数据")
        except Exception as e:
            print(f"加载芝加哥地图数据时出错: {str(e)}")
            chicago_data = {'cbgs': {}}
        
        # 分析不同阈值下的热点
        thresholds = [0.2, 0.3, 0.4]
        
        for threshold in thresholds:
            print(f"\n分析阈值 {threshold}:")
            
            # 查找LLM预测对但baseline预测错的CBG
            llm_correct_base_wrong = find_llm_correct_baseline_wrong(
                cot_results, base_results, ground_truth, threshold)
            
            if not llm_correct_base_wrong:
                print(f"在阈值 {threshold} 下没有找到LLM预测对但Baseline预测错的CBG")
                continue
                
            # 创建阈值对应的输出目录
            threshold_dir = os.path.join(OUTPUT_DIR, f"threshold_{int(threshold*100)}")
            os.makedirs(threshold_dir, exist_ok=True)
            
            # 为每个CBG生成Agent流向可视化
            for i, cbg_id in enumerate(llm_correct_base_wrong):
                print(f"处理CBG {cbg_id} ({i+1}/{len(llm_correct_base_wrong)})...")
                
                # 获取CBG描述
                cbg_description = get_cbg_description(cbg_id, CBG_META_PATH)
                
                # 收集LLM和Baseline的agent流向数据
                print(f"收集LLM模型访问CBG {cbg_id}的agent流向...")
                llm_agent_flows = collect_agent_flows(cbg_id, COT_INDIVIDUAL_DIR)
                
                print(f"收集Baseline模型访问CBG {cbg_id}的agent流向...")
                base_agent_flows = collect_agent_flows(cbg_id, BASE_INDIVIDUAL_DIR)
                
                # 生成流向图
                output_prefix = os.path.join(threshold_dir, f"cbg_{cbg_id}")
                visualize_agent_flows(
                    cbg_id, llm_agent_flows, base_agent_flows, chicago_data, cbg_description, output_prefix)
            
            print(f"已完成阈值 {threshold} 下的所有Agent流向可视化")
            
        print("\n所有可视化任务已完成!")
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 