import pandas as pd
import json

# 从筛选结果中提取视频名和human_judgement为1的记录
filtered_videos = []
data = json.load(open('VRU/output/filtered_videos_analysis.json', 'r'))
for video in data:
    if video['human_judgement'] == 1:
        # 提取视频序号（去掉.mp4后缀和前导零）
        video_num = int(video['video_name'].replace('.mp4', ''))
        filtered_videos.append({
            'video_number': video_num,
            'accident_frame': video['accident_frame'],
            'scene_complexity': video['scores']['scene_complexity']
        })

# 创建DataFrame并按视频序号排序
df = pd.DataFrame(filtered_videos)
df = df.sort_values('video_number')

# 设置表格格式
pd.set_option('display.max_rows', None)
df.to_csv('VRU/output/filtered_videos.csv', index=False)
print("筛选后的视频列表已保存至 VRU/output/filtered_videos.csv")
