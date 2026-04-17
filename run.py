import argparse
import json
import os
import sys
import time
from datetime import datetime

from tqdm import tqdm

from config import Config
from pipeline import Pipeline
from agent import Memory, Planner, Executor, create_tool_registry


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCT Pipeline: Agent-based medical transcription processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input RUCT file or folder path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-t", "--task", required=True, help="Task description (e.g., 'denoise, correct and identify speaker')")
    parser.add_argument("--model", default="gpt-4o", help="LLM backbone model (default: gpt-4o)")
    parser.add_argument("--planner-model", default=None, help="Model for Planner (defaults to --model)")
    parser.add_argument("--domain", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Medical domain (1-5, default: 1)")
    parser.add_argument("--api-key", default=None, help="API key (or set SCT_API_KEY env var)")
    parser.add_argument("--base-url", default=None, help="API base URL (or set SCT_BASE_URL env var)")
    parser.add_argument("--resume", action="store_true", help="Skip already completed files")
    parser.add_argument("--chunk-size", type=int, default=50, help="Lines per processing chunk (default: 50)")
    return parser.parse_args()


def collect_input_files(input_path: str) -> list:
    """Collect .txt and .ruct files from input path."""
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith((".txt", ".ruct"))
        )
    return [input_path]


def load_batch_log(log_path: str) -> dict:
    """Load existing batch log for resume support."""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_batch_log(log_path: str, log: dict):
    """Save batch log to disk."""
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    config = Config(
        input_path=args.input,
        output_dir=args.output,
        domain=args.domain,
        model_name=args.model,
        planner_model=args.planner_model or args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        chunk_size=args.chunk_size,
    )
    print(f"[Config] {config}")

    memory = Memory()

    pipeline = Pipeline(config, memory=memory)
    registry = create_tool_registry(pipeline)

    try:
        memory.write_long("domain_prompt", pipeline.get_domain_prompt())
    except Exception as e:
        print(f"[Warning] Could not load domain prompt: {e}")
    try:
        memory.write_long("knowledge", pipeline.load_knowledge())
    except Exception as e:
        print(f"[Warning] Could not load knowledge base: {e}")

    input_files = collect_input_files(args.input)
    print(f"[Agent] Found {len(input_files)} file(s) to process.")

    log_path = os.path.join(args.output, "batch_log.json")
    batch_log = load_batch_log(log_path)

    if args.resume and batch_log:
        completed = {f for f, info in batch_log.items() if info.get("status") in ("success", "partial")}
        before = len(input_files)
        input_files = [f for f in input_files if os.path.basename(f) not in completed]
        print(f"[Agent] Resume mode: skipping {before - len(input_files)} completed files.")

    if not input_files:
        print("[Agent] No files to process.")
        return

    summary = []
    for file_path in tqdm(input_files, desc="[Batch]", ncols=100):
        file_name = os.path.basename(file_path)
        output_path = os.path.join(args.output, file_name)
        steps_path = os.path.join(args.output, os.path.splitext(file_name)[0] + "_steps.json")
        start_time = datetime.now()

        try:
            memory.clear_short_term()

            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read()
            memory.write_short("input", input_text)

            batch_log[file_name] = {"status": "running", "start": start_time.isoformat()}
            save_batch_log(log_path, batch_log)

            planner = Planner(pipeline, model_name=config.planner_model)
            executor = Executor(pipeline, registry, memory, planner)

            plan = planner.create_plan(args.task, memory=memory)
            print(f"\n[Planner] Plan for '{file_name}':")
            for step in plan:
                print(f"  -> {step.get('name', '?')}: {step.get('tool', '?')}")

            t0 = time.time()
            result = executor.execute_plan(plan)
            duration = round(time.time() - t0, 2)

            with open(steps_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            log_detail_path = os.path.join(args.output, os.path.splitext(file_name)[0] + "_execution_log.json")
            with open(log_detail_path, "w", encoding="utf-8") as f:
                json.dump(result.get("execution_log", []), f, indent=2, ensure_ascii=False)

            final = result.get("final")
            has_errors = any(s.get("error") for s in result.get("steps", []))
            has_warnings = any(s.get("validation_warning") for s in result.get("steps", []))
            final_empty = final is None or (isinstance(final, str) and not final.strip())

            if has_errors or final_empty:
                error_msg = next(
                    (s.get("error") or s.get("validation_warning")
                     for s in result.get("steps", [])
                     if s.get("error") or s.get("validation_warning")),
                    "Empty or invalid output",
                )
                batch_log[file_name] = {
                    "status": "failed",
                    "error": error_msg,
                    "start": start_time.isoformat(),
                    "end": datetime.now().isoformat(),
                    "duration_sec": duration,
                }
                save_batch_log(log_path, batch_log)
                summary.append({"file": file_name, "status": f"failed: {error_msg}", "duration_sec": duration})
                print(f"[Agent] Failed '{file_name}': {error_msg}")
                continue

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final)

            status = "partial" if has_warnings else "success"
            warning_msgs = [
                s["validation_warning"] for s in result.get("steps", []) if s.get("validation_warning")
            ]

            batch_log[file_name] = {
                "status": status,
                "start": start_time.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_sec": duration,
                "output": output_path,
            }
            if warning_msgs:
                batch_log[file_name]["warnings"] = warning_msgs
            save_batch_log(log_path, batch_log)

            summary.append({"file": file_name, "status": status, "duration_sec": duration})
            if has_warnings:
                print(f"[Agent] Partial '{file_name}' (warnings: {warning_msgs})")
            print(f"[Agent] Completed '{file_name}' in {duration}s")

        except Exception as e:
            batch_log[file_name] = {
                "status": "failed",
                "error": str(e),
                "start": start_time.isoformat(),
                "end": datetime.now().isoformat(),
            }
            save_batch_log(log_path, batch_log)
            summary.append({"file": file_name, "status": f"failed: {e}"})
            print(f"[Agent] Error processing '{file_name}': {e}")

    report_path = os.path.join(args.output, "summary_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Agent] Done. Summary: {report_path}")


if __name__ == "__main__":
    main()
