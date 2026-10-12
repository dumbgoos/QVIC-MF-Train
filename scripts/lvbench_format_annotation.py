# Format LVBench video_info.meta.jsonl into one-QA-per-line JSONL for
# qvic/eval/model_video_lvbench.py (same as upstream FujitsuResearch/QViC-MF).
#
# Official meta nests many QA pairs under each video. The eval loop expects:
#   key = "{youtube_id}_{uid}", question, answer, question_type, ...

import argparse
import json


def format_lvbench_annotation(input_path: str, output_path: str) -> int:
    n = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            video_key = data["key"]
            video_type = data.get("type")
            video_info = data.get("video_info")
            for qa_pair in data["qa"]:
                formatted_entry = {
                    "key": f"{video_key}_{qa_pair['uid']}",
                    "type": video_type,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "question_type": qa_pair["question_type"],
                    "time_reference": qa_pair.get("time_reference"),
                    "video_info": video_info,
                }
                f_out.write(json.dumps(formatted_entry) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format LVBench annotations")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to LVBench video_info.meta.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to formatted one-QA-per-line JSONL",
    )
    args = parser.parse_args()
    n = format_lvbench_annotation(args.input, args.output)
    print(f"Wrote {n} QA lines to {args.output}")
