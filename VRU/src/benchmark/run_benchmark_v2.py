import os
import csv
import json
import random
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime
from models.model_zoo import get_model_list, get_model_runner

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[2]
MULTI_VERSION_DATA_DIR = PROJECT_ROOT / 'VRU' / 'vid_list' / 'multi_version_data'


def _split_env_csv(value: str):
    return [item.strip() for item in value.split(',') if item.strip()]


default_csv_list = [
    MULTI_VERSION_DATA_DIR / 'generated_options_2opts_20260327_085451.csv',
    MULTI_VERSION_DATA_DIR / 'generated_options_3opts_20260327_085451.csv',
    MULTI_VERSION_DATA_DIR / 'generated_options_4opts_20260327_085451.csv',
    MULTI_VERSION_DATA_DIR / 'generated_options_5opts_20260327_085451.csv',
]
qa_csv_env = os.getenv('BENCHMARK_QA_CSV_LIST', '').strip()
if qa_csv_env:
    QA_CSV_LIST = [Path(p) for p in _split_env_csv(qa_csv_env)]
else:
    QA_CSV_LIST = default_csv_list

RESULTS_DIR = Path(os.getenv('BENCHMARK_RESULTS_DIR', str(BENCHMARK_DIR / 'results')))
VIDEO_DIR = Path(os.getenv('BENCHMARK_VIDEO_DIR', str(PROJECT_ROOT / 'data' / 'crash' / 'videos' / 'Crash-1500')))
MAX_VIDEOS = int(os.getenv('BENCHMARK_MAX_VIDEOS', '0'))

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



def load_questions_from_csv(csv_path):
    questions_per_video = []
    encodings_to_try = ['utf-8-sig', 'utf-8', 'gb18030', 'cp1252', 'latin1']
    last_error = None

    for encoding in encodings_to_try:
        try:
            with open(csv_path, encoding=encoding) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                # 新格式: 每行一题 (video_id/question_id/question/correct_answer/option_n)
                if 'video_id' in fieldnames and 'question' in fieldnames and 'correct_answer' in fieldnames:
                    video_to_qas = {}
                    for row in reader:
                        video_number = (row.get('video_id') or '').strip()
                        q_text = (row.get('question') or '').strip()
                        correct_answer = (row.get('correct_answer') or '').strip()

                        if not video_number or not q_text or not correct_answer:
                            continue

                        options = [correct_answer]
                        option_idx = 1
                        while True:
                            option_key = f'option_{option_idx}'
                            if option_key not in row:
                                break
                            wrong_opt = (row.get(option_key) or '').strip()
                            if wrong_opt:
                                options.append(wrong_opt)
                            option_idx += 1

                        original_correct_idx = 0
                        idx_list = list(range(len(options)))
                        random.shuffle(idx_list)
                        shuffled_options = [options[j] for j in idx_list]
                        correct_index = idx_list.index(original_correct_idx) + 1

                        qa = {
                            'question': q_text,
                            'options': shuffled_options,
                            'correct_index': correct_index,
                            'question_id': int(row.get('question_id') or 0)
                        }
                        video_to_qas.setdefault(video_number, []).append(qa)

                    # 保持题目顺序稳定
                    for video_number in sorted(video_to_qas.keys()):
                        qas = sorted(video_to_qas[video_number], key=lambda x: x.get('question_id', 0))
                        for qa in qas:
                            qa.pop('question_id', None)
                        questions_per_video.append({'video_number': video_number, 'qas': qas})

                # 旧格式: 每行一个视频，列为 q1_text/q1_ans_correct/q1_ans_wrongk
                else:
                    for row in reader:
                        video_number = row['video_number']
                        qas = []
                        for i in range(1, 7):
                            q_text = row.get(f'q{i}_text', None)
                            if not q_text:
                                continue
                            options = []
                            answer = row.get(f'q{i}_ans_correct', None)
                            if answer:
                                options.append(answer)
                            for k in range(1, 6):
                                wrong_key = f'q{i}_ans_wrong{k}'
                                wrong_opt = row.get(wrong_key, None)
                                if wrong_opt and wrong_opt.strip():
                                    options.append(wrong_opt)
                            if not options:
                                continue
                            original_correct_idx = 0
                            idx_list = list(range(len(options)))
                            random.shuffle(idx_list)
                            shuffled_options = [options[j] for j in idx_list]
                            correct_index = idx_list.index(original_correct_idx) + 1
                            qas.append({
                                'question': q_text,
                                'options': shuffled_options,
                                'correct_index': correct_index
                            })
                        questions_per_video.append({'video_number': video_number, 'qas': qas})

            if encoding != 'utf-8-sig':
                print(f"  ℹ️ CSV 编码回退成功: {encoding} ({csv_path})")
            break
        except UnicodeDecodeError as e:
            last_error = e
            questions_per_video = []
            continue

    if not questions_per_video and last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f"CSV 解码失败，尝试编码: {encodings_to_try}，原始错误: {last_error.reason}"
        )

    if MAX_VIDEOS > 0:
        questions_per_video = questions_per_video[:MAX_VIDEOS]

    return questions_per_video


def build_single_question_prompt(qa_index, qa):
    """构造单题 prompt - 逐题提问模式
    
    Args:
        qa_index: 题目索引（1-based）
        qa: 单个问题字典 {'question': ..., 'options': [...], 'correct_index': ...}
    
    Returns:
        格式化的单题 prompt 字符串
    """
    q_text = qa['question'] if qa['question'] else ""
    num_options = len(qa.get('options', []))
    if num_options <= 0:
        num_options = 4
    option_numbers = ', '.join(str(i) for i in range(1, num_options + 1))
    
    prompt = f"Answer this multiple choice question. Respond with ONLY one number from: {option_numbers}. No explanation.\n\n"
    prompt += f"Question {qa_index}: {q_text}\n"
    
    for opt_idx, opt in enumerate(qa['options'], 1):
        prompt += f"  {opt_idx}. {opt}\n"
    
    prompt += f"\nYour answer (only the number): "
    
    return prompt


def run_model_inference(model_name, questions_per_video):
    """单个模型的推理流程 - 改为逐题提问"""
    try:
        print(f"\n{'='*70}")
        print(f"🚀 开始运行: {model_name} [开始时间: {datetime.now().strftime('%H:%M:%S')}]")
        print(f"{'='*70}")
        
        runner = get_model_runner(model_name)
        results = []
        
        for item in questions_per_video:
            video_number = item['video_number']
            qas = item['qas']
            
            # 加载实际视频帧（只加载一次）
            video_path = VIDEO_DIR / f"{video_number}.mp4"
            if not video_path.exists():
                print(f"  ⚠️ 视频不存在: {video_path}，跳过")
                continue
            
            print(f"  📹 {model_name}: 加载视频 {video_number}", end=" ... ")
            video_frames = load_video_frames(str(video_path), max_frames=32, target_fps=5)
            print(f"({len(video_frames)} 帧)")
            
            # 逐题推理
            choices = []
            for qa_idx, qa in enumerate(qas, 1):
                # 为每个题目构造单独的 prompt
                single_prompt = build_single_question_prompt(qa_idx, qa)
                num_options = len(qa.get('options', []))
                if num_options <= 0:
                    num_options = 4
                
                # 推理单个问题（传入相同的视频帧）
                try:
                    choice_list = runner.predict(video_number, single_prompt, video_frames, num_options=num_options)
                except TypeError:
                    # 兼容尚未支持 num_options 参数的旧版 runner
                    choice_list = runner.predict(video_number, single_prompt, video_frames)
                
                # 提取第一个数字作为答案
                answer = choice_list[0] if choice_list else 0
                choices.append(answer)
            
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
        
        # 结果文件名包含输入csv basename，防止覆盖
        if hasattr(run_model_inference, 'current_csv_path'):
            csv_basename = os.path.splitext(os.path.basename(run_model_inference.current_csv_path))[0]
            result_file = RESULTS_DIR / f'results_{model_name}-single_qa-{csv_basename}.json'
        else:
            result_file = RESULTS_DIR / f'results_{model_name}-single_qa.json'
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



print(f"\n{'='*70}")
print(f"📊 Video QA Benchmark 系统 (多CSV批量模式)")
print(f"{'='*70}")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"视频目录: {VIDEO_DIR}")
print(f"结果目录: {RESULTS_DIR}")
if MAX_VIDEOS > 0:
    print(f"小规模模式: 每个CSV只跑前 {MAX_VIDEOS} 个视频")

model_list = get_model_list()
print(f"\n配置的模型 ({len(model_list)}):")
for m in model_list:
    print(f"  - {m}")

for csv_path in QA_CSV_LIST:
    csv_path = str(csv_path)
    print(f"\n➡️ 处理VQA文件: {csv_path}")

    questions_per_video = load_questions_from_csv(csv_path)
    # 传递当前csv路径给run_model_inference
    run_model_inference.current_csv_path = csv_path
    if len(model_list) == 1:
        print(f"  串行模式（单模型）")
        result = run_model_inference(model_list[0], questions_per_video)
        summary = [result]
    else:
        print(f"  并行模式（{len(model_list)} 个模型，显存充足）")    
        threads = []
        results_dict = {}
        lock = threading.Lock()
        def thread_wrapper(model_name):
            run_model_inference.current_csv_path = csv_path
            result = run_model_inference(model_name, questions_per_video)
            with lock:
                results_dict[model_name] = result
        for model_name in model_list:
            thread = threading.Thread(target=thread_wrapper, args=(model_name,), daemon=False)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        summary = [results_dict[m] for m in model_list]

    print(f"\n{'='*70}")
    print(f"📈 运行总结 (文件: {csv_path})")
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
    print(f"\n💾 结果已保存到: {RESULTS_DIR}/")
    print(f"✅ {csv_path} 处理完成！")