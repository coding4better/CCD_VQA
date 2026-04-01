import os
import csv
import json
import random
import numpy as np
import threading
import time
from datetime import datetime
from models.model_zoo import get_model_list, get_model_runner

# QA_CSV = '/home/24068286g/UString/VRU/src/option_generate/data/QA_pair_v2_4options.csv'
QA_CSV = '/home/24068286g/UString/VRU/src/option_generate/data/QA_pair_v1_3options.csv'
# QA_CSV = '/home/24068286g/UString/VRU/src/option_generate/data/QA_pair_v3_5options.csv'
RESULTS_DIR = '/home/24068286g/UString/VRU/src/benchmark/results'
VIDEO_DIR = '/home/24068286g/UString/data/crash/videos/Crash-1500'

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_video_frames(video_path: str, max_frames: int = 50, target_fps: float = 2.0):
    """加载视频并均匀采样帧（不做亮度检测）"""
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ⚠️ 无法打开视频: {video_path}")
            return np.zeros((1, 480, 640, 3), dtype=np.uint8)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0 or fps <= 0:
            cap.release()
            return np.zeros((1, 480, 640, 3), dtype=np.uint8)
        frame_interval = max(1, int(fps / target_fps))
        frames = []
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
        cap.release()
        if not frames:
            return np.zeros((1, 480, 640, 3), dtype=np.uint8)
        return np.array(frames, dtype=np.uint8)
    except Exception as e:
        print(f"  ❌ 加载视频出错: {e}")
        return np.zeros((1, 480, 640, 3), dtype=np.uint8)


# 读取csv，每行一个视频，包含6个选择题
questions_per_video = []
with open(QA_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_number = row['video_number']
        qas = []
        for i in range(1, 7):
            q_text = row.get(f'q{i}_text', None)
            if not q_text:
                continue
            # 自动收集所有选项字段
            options = []
            answer = row.get(f'q{i}_ans_correct', None)
            if answer:
                options.append(answer)
            # 收集所有q{i}_ans_wrong*字段
            for k in range(1, 6):  # 最多支持5个选项
                wrong_key = f'q{i}_ans_wrong{k}'
                wrong_opt = row.get(wrong_key, None)
                if wrong_opt and wrong_opt.strip():
                    options.append(wrong_opt)

            # 选项随机打乱，追踪正确答案的新索引
            if not options:
                continue
            original_correct_idx = 0  # answer 已放在首位
            idx_list = list(range(len(options)))
            random.shuffle(idx_list)
            shuffled_options = [options[j] for j in idx_list]
            correct_index = idx_list.index(original_correct_idx) + 1  # 1-based

            qas.append({
                'question': q_text,
                'options': shuffled_options,
                'correct_index': correct_index
            })
        questions_per_video.append({'video_number': video_number, 'qas': qas})

# 构造prompt
def build_prompt(video_number, qas, max_question_len=80, max_option_len=60):
    """构造选择题prompt，避免超过 token 限制
    
    Args:
        video_number: 视频编号
        qas: 问题列表
        max_question_len: 问题最大长度
        max_option_len: 选项最大长度
    
    Returns:
        格式化的 prompt 字符串（控制在 4096 token 以内）
    """
    num_questions = len(qas)
    
    # 平衡的提示词 - 给出关键指导但避免过度复杂
    prompt = f"""Analyze the video carefully. Answer these {num_questions} questions.
Each has exactly 3 options. Only respond with option numbers (1, 2, or 3).

Pay attention to: movements, objects, scene details, timing.

"""
    
    for idx, qa in enumerate(qas, 1):
        q_text = qa['question'] if qa['question'] else ""
        prompt += f"Q{idx}: {q_text}\n"
        for opt_idx, opt in enumerate(qa['options'], 1):
            opt_short = opt if opt else ""
            prompt += f"  {opt_idx}. {opt_short}\n"
    
    prompt += f"\nFormat: Q1:# Q2:# Q3:# Q4:# Q5:# Q6:#\nAnswers: "
    
    return prompt

def run_model_inference(model_name, questions_per_video):
    """单个模型的推理流程"""
    try:
        print(f"\n{'='*70}")
        print(f"🚀 开始运行: {model_name} [开始时间: {datetime.now().strftime('%H:%M:%S')}]")
        print(f"{'='*70}")
        
        runner = get_model_runner(model_name)
        results = []
        
        for item in questions_per_video:
            video_number = item['video_number']
            qas = item['qas']
            prompt = build_prompt(video_number, qas)
            
            # 加载实际视频帧
            video_path = os.path.join(VIDEO_DIR, f"{video_number}.mp4")
            if not os.path.exists(video_path):
                print(f"  ⚠️ 视频不存在: {video_path}，跳过")
                continue
            
            print(f"  📹 {model_name}: 加载视频 {video_number}", end=" ... ")
            video_frames = load_video_frames(video_path, max_frames=50, target_fps=2.0)
            print(f"({len(video_frames)} 帧)")
            
            # 推理（传入视频帧）
            choices = runner.predict(video_number, prompt, video_frames)
            
            # 计算正确答案的选项序号（已打乱后的索引，1-based）
            correct = [qas[i]['correct_index'] for i in range(len(qas))]
            
            # 计算准确率
            num_questions = len(qas)
            valid_choices = choices[:num_questions]
            correct_count = sum([c==r for c,r in zip(valid_choices, correct)])
            acc = correct_count / num_questions if num_questions > 0 else 0
            
            results.append({
                'video_number': video_number,
                'choices': valid_choices,
                'correct': correct,
                'accuracy': acc,
                'num_questions': num_questions,
                'correct_count': correct_count
            })
        
        # 计算模型整体准确率
        total_correct = sum([r['correct_count'] for r in results])
        total_questions = sum([r['num_questions'] for r in results])
        overall_acc = total_correct / total_questions if total_questions > 0 else 0
        
        # 保存结果
        results_summary = {
            'model_name': model_name,
            'overall_accuracy': overall_acc,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'num_videos': len(results),
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        result_file = f'{RESULTS_DIR}/results_{model_name}.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ {model_name} 完成:")
        print(f"  整体准确率: {overall_acc:.2%} ({total_correct}/{total_questions})")
        print(f"  视频数: {len(results)}")
        print(f"  结果已保存: {result_file}")
        print(f"  完成时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 释放资源
        runner.release()
        
        return {
            'model': model_name,
            'accuracy': overall_acc,
            'status': 'success'
        }
    
    except Exception as e:
        print(f"\n❌ {model_name} 执行失败:")
        print(f"  错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'error': str(e),
            'status': 'failed'
        }


# 主流程 - 支持并行运行
print(f"\n{'='*70}")
print(f"📊 Video QA Benchmark 系统")
print(f"{'='*70}")

model_list = get_model_list()
print(f"\n配置的模型 ({len(model_list)}):")
for m in model_list:
    print(f"  - {m}")

# 如果只有一个模型，串行运行；多个模型时并行运行
if len(model_list) == 1:
    print(f"\n➡️  串行模式（单模型）")
    result = run_model_inference(model_list[0], questions_per_video)
    summary = [result]
else:
    print(f"\n⚡ 并行模式（{len(model_list)} 个模型，显存充足）")    
    threads = []
    results_dict = {}
    lock = threading.Lock()
    
    def thread_wrapper(model_name):
        result = run_model_inference(model_name, questions_per_video)
        with lock:
            results_dict[model_name] = result
    
    # 启动所有模型的推理线程
    for model_name in model_list:
        thread = threading.Thread(target=thread_wrapper, args=(model_name,), daemon=False)
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    summary = [results_dict[m] for m in model_list]

# 输出总结
print(f"\n{'='*70}")
print(f"📈 运行总结")
print(f"{'='*70}\n")
print(f"{'模型':<25} {'准确率':<12} {'状态':<12}")
print("-" * 70)

for result in summary:
    if result['status'] == 'success':
        accuracy = result['accuracy']
        print(f"{result['model']:<25} {accuracy:.2%}         {'✓':<12}")
    else:
        print(f"{result['model']:<25} {'N/A':<12} {'❌ 失败':<12}")

print(f"{'='*70}")
print(f"\n💾 所有结果已保存到: {RESULTS_DIR}/")
print(f"✅ Benchmark 运行完成！")

