import pandas as pd
import re
from pathlib import Path

# 1. 重新读取并加载所有原始数据
# 原xlsx文件（含完整标注信息）
xlsx_df = pd.read_excel("E:\\study\\project\\CCD_VQA\\VRU\\vid_list\\100_anotated_videos.xlsx")
# list文件（用于提取筛选条件）
csv_df = pd.read_csv('E:\\study\\project\\CCD_VQA\\VRU\\vid_list\\video_list_comparison_3cols.csv')

# 2. 重复提取list第二列的有效视频编号（确保筛选条件一致）
def extract_video_number(filename):
    if pd.isna(filename):
        return None
    match = re.search(r'(\d+)\.mp4', str(filename))
    return int(match.group(1)) if match else None

# 提取并过滤有效编号
csv_second_col = csv_df['txt_version_files']
valid_video_numbers = [num for num in map(extract_video_number, csv_second_col) if num is not None]
print(f"从list第二列提取的有效视频编号总数：{len(valid_video_numbers)}")

# 3. 扫描原xlsx，筛选匹配的行（关键步骤）
# 筛选条件：xlsx的video_number在有效编号列表中
matched_rows = xlsx_df[xlsx_df['video_number'].isin(valid_video_numbers)].copy()

# 3.1 将txt_version_files中未匹配到xlsx的编号追加到首列（其余列为空）
valid_video_numbers_unique = sorted(set(valid_video_numbers))
matched_video_numbers = set(matched_rows['video_number'].tolist())
missing_numbers = [num for num in valid_video_numbers_unique if num not in matched_video_numbers]

if missing_numbers:
    missing_rows = pd.DataFrame({col: [pd.NA] * len(missing_numbers) for col in matched_rows.columns})
    missing_rows['video_number'] = missing_numbers
    matched_rows = pd.concat([matched_rows, missing_rows], ignore_index=True)

# 3.2 对所有行按视频编号排序
matched_rows = matched_rows.sort_values(by='video_number', ascending=True).reset_index(drop=True)

# 4. 数据匹配结果统计
print(f"\n原xlsx文件总行数：{len(xlsx_df)}")
print(f"新表格总行数（含补齐行）：{len(matched_rows)}")
print(f"补齐的编号数量：{len(missing_numbers)}")

# 显示匹配到的视频编号（前10个），方便核对
print(f"\n排序后前10个视频编号：{list(matched_rows['video_number'].head(10))}")

# 5. 输出到目标路径 VRU\vid_list
output_dir = Path(r"E:\study\project\CCD_VQA\VRU\vid_list")
output_dir.mkdir(parents=True, exist_ok=True)

output_xlsx = output_dir / "matched_videos_from_txt_version.xlsx"
output_csv = output_dir / "matched_videos_from_txt_version.csv"

fallback_xlsx = output_dir / "matched_videos_from_txt_version_updated.xlsx"
fallback_csv = output_dir / "matched_videos_from_txt_version_updated.csv"

try:
    matched_rows.to_excel(output_xlsx, index=False)
    written_xlsx = output_xlsx
except PermissionError:
    matched_rows.to_excel(fallback_xlsx, index=False)
    written_xlsx = fallback_xlsx

try:
    matched_rows.to_csv(output_csv, index=False, encoding="utf-8-sig")
    written_csv = output_csv
except PermissionError:
    matched_rows.to_csv(fallback_csv, index=False, encoding="utf-8-sig")
    written_csv = fallback_csv

print(f"\n已输出Excel文件：{written_xlsx}")
print(f"已输出CSV文件：{written_csv}")
print(f"输出行数：{len(matched_rows)}")