import json
from pathlib import Path


def normalize_options(options):
    """Normalize options to list[{'key': 'A', 'value': '...'}]."""
    if options is None:
        return []

    if isinstance(options, dict):
        normalized = []
        for k, v in options.items():
            normalized.append({"key": str(k), "value": "" if v is None else str(v)})
        return normalized

    if isinstance(options, list):
        if not options:
            return []

        # Already list of dicts
        if all(isinstance(x, dict) for x in options):
            normalized = []
            for idx, opt in enumerate(options):
                key = opt.get("key")
                value = opt.get("value")
                if key is None:
                    key = chr(ord("A") + idx)
                normalized.append({"key": str(key), "value": "" if value is None else str(value)})
            return normalized

        # list of strings/others
        normalized = []
        for idx, opt in enumerate(options):
            key = chr(ord("A") + idx)
            normalized.append({"key": key, "value": "" if opt is None else str(opt)})
        return normalized

    return []


def resolve_answer_text(qa, normalized_options):
    """Resolve answer text from answer/correct_answer/correct_answer_index/options."""
    # Priority 1: answer field (can be string or integer index)
    answer = qa.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip(), "answer"
    
    # Priority 1b: answer is an integer index into options
    if isinstance(answer, int) and 0 <= answer < len(normalized_options):
        v = (normalized_options[answer].get("value") or "").strip()
        if v:
            return v, "answer"
        return "", "missing"

    # Priority 2: correct_answer_index (integer index into options)
    answer_index = qa.get("correct_answer_index")
    if isinstance(answer_index, int) and 0 <= answer_index < len(normalized_options):
        v = (normalized_options[answer_index].get("value") or "").strip()
        if v:
            return v, "correct_answer_index"
        return "", "missing"

    # Priority 3: correct_answer as text or key
    correct = qa.get("correct_answer")
    if isinstance(correct, str) and correct.strip():
        c = correct.strip()

        # If looks like key, map via options
        if c in ["A", "B", "C", "D", "E", "F"]:
            for opt in normalized_options:
                if opt.get("key") == c:
                    v = (opt.get("value") or "").strip()
                    if v:
                        return v, "correct_answer_key"
            return "", "correct_answer_key_not_found"

        # Otherwise treat as direct text
        return c, "correct_answer_text"

    return "", "missing"


def standardize_file(input_path: Path, output_path: Path, missing_report_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "total_records": len(data),
        "total_qa": 0,
        "answer_resolved": 0,
        "answer_missing": 0,
        "sources": {
            "answer": 0,
            "correct_answer_text": 0,
            "correct_answer_key": 0,
            "correct_answer_index": 0,
            "correct_answer_key_not_found": 0,
            "missing": 0,
        },
    }

    standardized = []
    missing_report = []

    for item in data:
        new_item = dict(item)
        vqa_list = item.get("generated_vqa", [])

        if not isinstance(vqa_list, list):
            vqa_list = []

        new_vqa = []
        for idx, qa in enumerate(vqa_list):
            stats["total_qa"] += 1

            if not isinstance(qa, dict):
                qa = {}

            question = qa.get("question", "")
            question = question.strip() if isinstance(question, str) else ""

            normalized_options = normalize_options(qa.get("options"))
            answer_text, source = resolve_answer_text(qa, normalized_options)
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

            if answer_text:
                stats["answer_resolved"] += 1
            else:
                stats["answer_missing"] += 1
                missing_report.append(
                    {
                        "video_name": item.get("video_name", ""),
                        "video_id": _extract_video_id(item.get("video_name", "")),
                        "qa_index": idx,
                        "question_id": qa.get("question_id", idx + 1),
                        "question": question,
                        "reason": source,
                    }
                )

            new_qa = {
                "id": qa.get("id", idx + 1),
                "question_id": qa.get("question_id", idx + 1),
                "dimension": qa.get("dimension", ""),
                "question": question,
                "options": normalized_options,
                "correct_answer": answer_text,
                "answer_source": source,
            }
            new_vqa.append(new_qa)

        new_item["generated_vqa"] = new_vqa
        standardized.append(new_item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(standardized, f, ensure_ascii=False, indent=2)

    with missing_report_path.open("w", encoding="utf-8") as f:
        json.dump(missing_report, f, ensure_ascii=False, indent=2)

    return stats


def _extract_video_id(video_name):
    if not isinstance(video_name, str):
        return None
    stem = Path(video_name).stem
    return int(stem) if stem.isdigit() else None


def main():
    input_path = Path("VRU/vid_list/generated_vqa_199.json")
    output_path = Path("VRU/vid_list/generated_vqa_199_standarized1.json")
    missing_report_path = Path("VRU/vid_list/generated_vqa_199_standarized_missing_report.json")

    stats = standardize_file(input_path, output_path, missing_report_path)

    print("Standardization complete")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Missing report: {missing_report_path}")
    print("---")
    print(f"Total records:      {stats['total_records']}")
    print(f"Total QA entries:   {stats['total_qa']}")
    print(f"Resolved answers:   {stats['answer_resolved']}")
    print(f"Missing answers:    {stats['answer_missing']}")
    print("Sources:")
    for k, v in stats["sources"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
