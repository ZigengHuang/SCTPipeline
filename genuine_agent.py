"""
assistant_agent1027.py  —  Agent wrapper for the medical transcription pipeline
Updated to match manuscript: Planner/Memory/Executor + Tool orchestration + validation/replanning.

Usage examples:
"""

import argparse
import traceback
import sys
import json
import os
import time
import io
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

# Import the existing pipeline (must be in same dir or installable module)
from medical_transcription_pipeline import Config, MedicalTranscriptionPipeline

# ---------- Memory module ----------
class Memory:
    """
    Tiny memory store for short-term (per-run) and long-term (persistent) memories.
    Short-term used for current task (intermediate step outputs), long-term for domain/knowledge config.
    """
    def __init__(self):
        self.short_term: Dict[str, Any] = {}
        self.long_term: Dict[str, Any] = {}

    def write_short(self, key: str, value: Any):
        self.short_term[key] = value

    def read_short(self, key: str, default=None):
        return self.short_term.get(key, default)

    def write_long(self, key: str, value: Any):
        self.long_term[key] = value

    def read_long(self, key: str, default=None):
        return self.long_term.get(key, default)

# ---------- Tools & Registry ----------
class Tool:
    def __init__(self, name: str, func: Callable[..., Any], description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self.tools[name]

    def list_tools(self):
        return {name: t.description for name, t in self.tools.items()}

# ---------- Planner ----------
class Planner:
    """
    Planner: try LLM-based planning (JSON list of steps), fallback to rule-based plan.
    Each step: {"action":"use_tool","tool":"denoise","args":["{ref:input}"],"name":"step1"}
    """
    def __init__(self, pipeline: MedicalTranscriptionPipeline, model_name: str = "gpt-5-nano", temperature: float = 0.0):
        self.pipeline = pipeline
        self.model_name = model_name
        self.temperature = temperature

    def _call_model_for_plan(self, task: str) -> List[Dict]:
        """
        Use the model to automatically generate a JSON plan describing
        which tools to call (and in what order) to complete the user's task.
        """

        prompt = f"""
    You are an intelligent planner for a medical text processing agent.
    Decompose the user task into a JSON list of tool calls in correct order.

    Available tools:
    - denoise: Remove noise and irrelevant artifacts from raw medical transcripts.
    - correction: Correct grammar, transcription, and formatting errors.
    - speaker_identify: Identify and label speakers (Doctor, Patient, Nurse, etc.).
    - segment_topic: Segment the conversation by topic changes (Topic-based segmentation).
    - segment_chrono: Segment the conversation by chronological order (timestamp-based segmentation).
    - segment_emotion: Segment the conversation by emotional transitions (Emotion-based segmentation).
    - sct_pipeline: Execute the full Structured Clinical Transcript pipeline.

    Each JSON element must include:
    {{
      "action": "use_tool",
      "tool": "<tool_name>",
      "args": ["<input_reference>"],
      "name": "<unique_step_name>"
    }}

    Guidelines:
    - Always output a pure JSON array, no extra text.
    - Reference the original input as "{{ref:input}}".
    - Reference previous steps using "{{ref:<step_name>}}".
    - Choose the segmentation tool (topic / chrono / emotion) based on user intent.
    - If the user only says "segment", use topic segmentation by default.

    User task: {task}
    """

        try:
            client = getattr(self.pipeline, "client", None)
            if client is None:
                raise RuntimeError("No client attached to pipeline for planning.")

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            # Robustly extract the model’s JSON output
            if hasattr(response, "choices") and len(response.choices) > 0:
                content = (
                    response.choices[0].message.content
                    if hasattr(response.choices[0], "message")
                    else response.choices[0].text
                )
            else:
                content = str(response)

            # Find first '[' to cut off any prefix text
            start = content.find('[')
            json_text = content[start:] if start != -1 else content

            plan = json.loads(json_text)
            if not isinstance(plan, list):
                raise ValueError("Planner LLM returned non-list JSON structure.")
            return plan

        except Exception as e:
            print("Planner LLM failed or returned invalid JSON. Falling back to rule-based planner.")
            print("Planner error:", str(e))
            return self._fallback_plan(task)

    def _fallback_plan(self, task: str) -> List[Dict]:
        t = task.lower()
        plan = []

        # common directives
        if "run pipeline" in t or "whole pipeline" in t or "run the pipeline" in t:
            plan = [{"action": "use_tool", "tool": "sct_pipeline", "args": ["{ref:input}"], "name": "run_all"}]
            return plan

        # heuristics: preprocess -> denoise -> correction -> speaker -> segmentation
        if "preprocess" in t or "pre-seg" in t or "clean format" in t:
            plan.append({"action": "use_tool", "tool": "preprocess", "args": ["{ref:input}"], "name": "preprocess"})

        if "denoise" in t or "noise" in t or "clean" in t:
            plan.append({"action": "use_tool", "tool": "denoise", "args": ["{ref:input}"], "name": "denoise"})

        if "correct" in t or "correction" in t or "fix" in t or "proofread" in t:
            prev = "{ref:denoise}" if any(p["tool"] == "denoise" for p in plan) else "{ref:input}"
            plan.append({"action": "use_tool", "tool": "correction", "args": [prev], "name": "correction"})

        if "speaker" in t or "identify speaker" in t or "speaker id" in t or "who said" in t:
            prev = "{ref:correction}" if any(p["tool"] == "correction" for p in plan) else "{ref:input}"
            plan.append({"action": "use_tool", "tool": "speaker_identify", "args": [prev], "name": "speaker_identify"})

        # segmentation logic (no interactive version)
        if "emotion" in t or "sentiment" in t or "feel" in t or "affect" in t:
            prev = "{ref:speaker_identify}" if any(p["tool"] == "speaker_identify" for p in plan) else "{ref:input}"
            plan.append({"action": "use_tool", "tool": "segment_emotion", "args": [prev], "name": "segment_emotion"})

        elif "chronolog" in t or "chronological" in t or "timeline" in t or "time order" in t or "time-based" in t:
            prev = "{ref:speaker_identify}" if any(p["tool"] == "speaker_identify" for p in plan) else "{ref:input}"
            plan.append({"action": "use_tool", "tool": "segment_chrono", "args": [prev], "name": "segment_chrono"})

        elif "topic" in t or "by topic" in t or "topic-based" in t or "segment" in t or "segmentation" in t:
            prev = "{ref:speaker_identify}" if any(p["tool"] == "speaker_identify" for p in plan) else "{ref:input}"
            plan.append({"action": "use_tool", "tool": "segment_topic", "args": [prev], "name": "segment_topic"})

        # sct pipeline if mentioned explicitly
        if "sct" in t or "structured clinical transcript" in t:
            prev = "{ref:segment_topic}" if any(p["tool"].startswith("segment_") for p in plan) else "{ref:input}"
            plan.append({"action": "use_tool", "tool": "sct_pipeline", "args": [prev], "name": "sct_pipeline"})

        # default fallback if nothing matched
        if not plan:
            plan = [
                {"action": "use_tool", "tool": "denoise", "args": ["{ref:input}"], "name": "denoise"},
                {"action": "use_tool", "tool": "correction", "args": ["{ref:denoise}"], "name": "correction"},
                {"action": "use_tool", "tool": "speaker_identify", "args": ["{ref:correction}"],
                 "name": "speaker_identify"},
                {"action": "use_tool", "tool": "segment_topic", "args": ["{ref:speaker_identify}"],
                 "name": "segment_topic"},
            ]

        print("[Fallback planner] Generated rule-based plan:")
        for step in plan:
            print(" -", step)
        return plan

    def create_plan(self, task: str) -> List[Dict]:
        plan = self._call_model_for_plan(task)
        if not isinstance(plan, list):
            raise ValueError("Planner returned unexpected structure.")
        return plan

# ---------- Agent Executor ----------
class Agent:
    def __init__(self, pipeline: MedicalTranscriptionPipeline, tool_registry: ToolRegistry, memory: Memory, planner: Planner):
        self.pipeline = pipeline
        self.tools = tool_registry
        self.memory = memory
        self.planner = planner
        self.step_outputs: Dict[str, Any] = {}

    def _resolve_arg(self, arg: Any):
        """Resolve {ref:name} tokens to actual data (step outputs or memory)."""
        if not isinstance(arg, str):
            return arg
        if arg.startswith("{ref:") and arg.endswith("}"):
            ref = arg[len("{ref:"):-1]
            if ref == "input":
                return self.memory.read_short("input")
            # step outputs prioritized over short-term memory
            return self.step_outputs.get(ref) or self.memory.read_short(ref)
        return arg

    def _validate_step(self, step: Dict, input_value: Any, output_value: Any) -> (bool, Optional[str]):
        """
        Basic validators for the main invariants from manuscript:
         - For line-by-line tools (denoise/correction/speaker_identify), keep same number of lines.
         - For segmentation tools, ensure output is non-empty.
        Returns (is_ok, message_if_not_ok)
        """
        tool = step.get("tool")
        # If input was string with lines
        try:
            if isinstance(input_value, str) and isinstance(output_value, str):
                in_lines = input_value.splitlines()
                out_lines = output_value.splitlines()
                # For denoise/correction/speaker_identify: must preserve SAME number of lines
                if tool in ("denoise", "correction", "speaker_identify"):
                    if len(in_lines) != len(out_lines):
                        return False, f"Line count mismatch for {tool}: in={len(in_lines)} out={len(out_lines)}"
            # segmentation: expect non-empty structured output
            if tool == "segmentation":
                if (not output_value) or (isinstance(output_value, str) and len(output_value.strip()) == 0):
                    return False, "Segmentation returned empty output"
        except Exception as e:
            return False, f"Validator error: {e}"
        return True, None

    def _attempt_repair(self, step: Dict, input_value: Any):
        """
        Try one or two simple repair strategies:
         - If pipeline provides chunk processors, use them (process_in_chunks).
         - Otherwise, re-run tool with same args but forced options where possible.
        Returns repaired_output or raises.
        """
        tool_name = step.get("tool")
        try:
            if hasattr(self.pipeline, "_process_in_chunks") and tool_name in ("correction", "denoise",):
                # call underlying chunked method where appropriate
                if tool_name == "correction" and hasattr(self.pipeline, "_correct_content"):
                    repaired = self.pipeline._process_in_chunks(self.pipeline._correct_content_chunk, input_value, chunk_size=50)
                    return repaired
                if tool_name == "denoise" and hasattr(self.pipeline, "_remove_noise"):
                    # If _remove_noise exists as chunk-capable wrapper, try in-chunks re-run
                    repaired = self.pipeline._process_in_chunks(self.pipeline._remove_noise, input_value, chunk_size=50)
                    return repaired
            # Generic fallback: call tool again with same args (may succeed)
            tool = self.tools.get(tool_name)
            resolved_args = [self._resolve_arg(a) for a in step.get("args", [])]
            if len(resolved_args) == 1:
                return tool.run(resolved_args[0])
            else:
                return tool.run(*resolved_args)
        except Exception as e:
            raise RuntimeError(f"Repair attempt failed for {tool_name}: {e}")

    def _run_step(self, step: Dict) -> Dict:
        if step.get("action") != "use_tool":
            raise ValueError(f"Unsupported action: {step.get('action')}")
        tool_name = step.get("tool")
        args = step.get("args", [])
        step_name = step.get("name") or f"step_{len(self.step_outputs)+1}"
        # resolve args
        resolved_args = [self._resolve_arg(a) for a in args]
        tool = self.tools.get(tool_name)
        print(f"[Agent] Executing '{step_name}' -> tool '{tool_name}' with args types: {[type(a).__name__ for a in resolved_args]}")
        try:
            # call tool; if tool expects (text) and we passed list etc, attempt safe mapping
            if len(resolved_args) == 0:
                result = tool.run()
            elif len(resolved_args) == 1:
                result = tool.run(resolved_args[0])
            else:
                result = tool.run(*resolved_args)
        except TypeError:
            # try passing the first arg as string if TypeError
            if resolved_args:
                result = tool.run(resolved_args[0])
            else:
                raise
        except Exception as e:
            print(f"[Agent] Error running tool {tool_name}: {e}")
            raise

        # Save result into step_outputs and short-term memory
        self.step_outputs[step_name] = result
        self.memory.write_short(step_name, result)

        # Validate
        input_for_validation = resolved_args[0] if resolved_args else None
        ok, msg = self._validate_step(step, input_for_validation, result)
        if not ok:
            print(f"[Agent] Validation failed after step {step_name}: {msg}")
            # attempt local repair/retry once
            try:
                repaired = self._attempt_repair(step, input_for_validation)
                # re-validate repaired
                ok2, msg2 = self._validate_step(step, input_for_validation, repaired)
                if ok2:
                    print(f"[Agent] Repair succeeded for step {step_name}")
                    self.step_outputs[step_name] = repaired
                    self.memory.write_short(step_name, repaired)
                    return {"step": step_name, "tool": tool_name, "result": repaired, "repaired": True}
                else:
                    print(f"[Agent] Repair attempted but still invalid: {msg2}")
            except Exception as e:
                print(f"[Agent] Repair exception: {e}")
            # If still invalid, trigger replanning: ask planner for new plan targeted to fix this tool
            # We create a targeted task description and request a new plan
            repair_task = f"Repair the output of tool {tool_name} for the input. Try alternative strategies preserving invariants."
            new_plan = self.planner._fallback_plan(repair_task)
            print(f"[Agent] Triggering fallback replanning with plan: {new_plan}")
            # execute fallback plan steps (simple immediate execution)
            fallback_results = []
            for sub in new_plan:
                try:
                    out = self._run_step(sub)
                    fallback_results.append(out)
                except Exception as e:
                    print(f"[Agent] Fallback sub-step failed: {e}")
                    break
            # After fallback attempts, return original result but annotate failure
            return {"step": step_name, "tool": tool_name, "result": result, "validation_error": msg, "fallback_results": fallback_results}
        return {"step": step_name, "tool": tool_name, "result": result}

    def execute_plan(self, plan: List[Dict]) -> Dict:
        results = []
        for step in plan:
            try:
                out = self._run_step(step)
                results.append(out)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Agent] Execution stopped due to error at step {step.get('name')}: {e}")
                results.append({"step": step.get("name"), "tool": step.get("tool"), "error": str(e), "traceback": tb})
                break
        final_output = results[-1].get("result") if results else None
        return {"steps": results, "final": final_output, "all_step_outputs": self.step_outputs}

# ---------- Tool wrappers mapping to pipeline's methods ----------
def create_tools_from_pipeline(pipeline: MedicalTranscriptionPipeline) -> ToolRegistry:
    registry = ToolRegistry()

    # Preprocess -> returns list of segments (call pipeline.preprocess_ruct)
    # registry.register(Tool(
    #     "preprocess",
    #     func=lambda text_or_path=None: pipeline.preprocess_ruct(),
    #     description="Pre-segment input RUCT into smaller segments (uses pipeline.cfg.input_ruct)."
    # ))

    # denoise -> wrap _remove_noise which expects text
    registry.register(Tool(
        "denoise",
        func=lambda text: pipeline._remove_noise(text),
        description="Remove noise from a text segment (preserve line count)."
    ))

    # correction -> wrap _correct_content
    registry.register(Tool(
        "correction",
        func=lambda text: pipeline._correct_content(text),
        description="Apply domain-aware content correction (preserve line count)."
    ))

    # speaker_identify
    registry.register(Tool(
        "speaker_identify",
        func=lambda text: pipeline._identify_speakers(text),
        description="Identify speakers line-by-line and label as Doctor/Patient/Others."
    ))

    # segmentation wrapper -> expects text & optional strategy
    # registry.register(Tool(
    #     "segmentation",
    #     func=lambda text, strategy="Topic-based": pipeline.apply_segmentation(text, strategy),
    #     description="Segment processed text; default strategy: Topic-based."
    # ))
    registry.register(
        Tool(
            "segment_topic",
            func=lambda text: pipeline.apply_segmentation(text, "Topic-based"),
            description="Segment the conversation by topic changes (Topic-based)."
        )
    )
    registry.register(
        Tool(
            "segment_chrono",
            func=lambda text: pipeline.apply_segmentation(text, "Chronological"),
            description="Segment the conversation by chronological order (Chronological)."
        )
    )
    registry.register(
        Tool(
            "segment_emotion",
            func=lambda text: pipeline.apply_segmentation(text, "Emotion-based"),
            description="Segment the conversation by emotional transitions (Emotion-based)."
        )
    )


    # sct_pipeline -> run end-to-end pipeline on input segments (auto)
    def sct_pipeline_auto(input_text):
        """
        Safe wrapper:
          - If input_text is a list -> process each segment and assemble output.
          - If string -> treat as single segment.
          - Load knowledge and domain prompt into memory for consistent behavior.
        """
        # ensure knowledge is loaded to pipeline and return assembled result
        try:
            knowledge = pipeline.load_knowledge()
        except Exception as e:
            print("[Tool:sct_pipeline] Warning: load_knowledge() failed or missing files:", e)
            knowledge = None

        # get domain prompt
        try:
            domain_prompt = pipeline.get_domain_prompt()
        except Exception as e:
            domain_prompt = None

        # If passed a single string treat as single segment
        if isinstance(input_text, list):
            segments = input_text
        else:
            segments = [input_text]

        # process each segment via pipeline.process_segment (assumes it exists)
        processed = []
        for seg in segments:
            processed.append(pipeline.process_segment(seg))
        assembled = pipeline.generate_output(processed)
        return assembled

    registry.register(Tool(
        "sct_pipeline",
        func=sct_pipeline_auto,
        description="Run an automated pipeline on provided segments or text; returns assembled SCT."
    ))

    return registry

# ---------- CLI Entrypoint ----------
def main():
    import io
    import sys
    import json
    import time
    import argparse
    import os
    from tqdm import tqdm
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Agent wrapper for medical transcription pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input file or folder path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--task", "-t", required=True, help="Task description (e.g. 'Please denoise, correct and identify speaker this clinical note')")
    parser.add_argument("--planner-model", default="gpt-5-nano", help="Model name for planner (default gpt-5-nano)")
    parser.add_argument("--resume", action="store_true", help="Resume mode: skip already completed files based on batch_log.json")
    parser.add_argument("--domain", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Medical domain index (1-5). Default=1")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, "batch_log.json")
    report_path = os.path.join(args.output, "summary_report.txt")

    # 加载日志文件（如果存在）
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                batch_log = json.load(f)
        except Exception:
            batch_log = {}
    else:
        batch_log = {}

    # 自动 domain=1 当输入为文件夹
    if os.path.isdir(args.input):
        args.domain = 1

    print(f"[Agent] Processing input: {args.input}")
    print(f"Debug: Input path resolved to {os.path.abspath(args.input)}")
    print(f"Debug: Output directory resolved to {os.path.abspath(args.output)}")

    # 初始化 Config（自动喂 domain）
    original_argv = sys.argv.copy()
    original_stdin = sys.stdin
    try:
        sys.argv = [original_argv[0], "-i", args.input, "-o", args.output]
        fake_input = io.StringIO(f"{args.domain}\n")
        sys.stdin = fake_input
        config = Config()
        sys.stdin = original_stdin

        try:
            config.medical_domain, config.domain_key, config.knowledge_base = config.domain_mapping[args.domain]
            config.knowledge_base = os.path.abspath(config.knowledge_base)
        except Exception:
            pass
    finally:
        sys.argv = original_argv
        sys.stdin = original_stdin

    pipeline = MedicalTranscriptionPipeline(config)
    registry = create_tools_from_pipeline(pipeline)
    memory = Memory()

    # 支持文件夹或单文件
    if os.path.isdir(args.input):
        input_files = [
            os.path.join(args.input, fn)
            for fn in os.listdir(args.input)
            if fn.endswith(".txt") or fn.endswith(".ruct")
        ]
        print(f"[Agent] Detected folder input with {len(input_files)} files.")
    else:
        input_files = [args.input]

    # 智能 resume-by-log 模式
    if args.resume and batch_log:
        to_skip = [f for f, info in batch_log.items() if info.get("status") == "success"]
        input_files = [
            f for f in input_files
            if os.path.basename(f) not in to_skip
        ]
        print(f"[Agent] Resume mode: skipping {len(to_skip)} completed files based on batch_log.json")

    # 批量执行 + tqdm 进度条 + 实时日志
    summary = []
    for file_path in tqdm(input_files, desc="[Batch Progress]", ncols=100):
        file_name = os.path.basename(file_path)
        output_path = os.path.join(args.output, file_name)
        steps_path = os.path.join(args.output, os.path.splitext(file_name)[0] + "_steps.json")
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 防止重复写入 log
        if batch_log.get(file_name, {}).get("status") == "success":
            continue

        try:
            print(f"\n[Agent] Processing file: {file_name}")
            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read()
            memory.write_short("input", input_text)

            # 写日志（开始状态）
            batch_log[file_name] = {"status": "running", "start": start_time}
            with open(log_path, "w", encoding="utf-8") as lf:
                json.dump(batch_log, lf, indent=2, ensure_ascii=False)

            # 知识加载
            try:
                kp = pipeline.get_domain_prompt()
                memory.write_long("domain_prompt", kp)
            except Exception:
                pass
            try:
                kn = pipeline.load_knowledge()
                memory.write_long("knowledge", kn)
            except Exception:
                pass

            planner = Planner(pipeline, model_name=args.planner_model)
            agent = Agent(pipeline, registry, memory, planner)

            print("[Agent] Creating plan for task:", args.task)
            plan = planner.create_plan(args.task)
            print("[Agent] Plan created:", json.dumps(plan, ensure_ascii=False, indent=2))

            print("[Agent] Executing plan...")
            t0 = time.time()
            execution_result = agent.execute_plan(plan)
            duration = round(time.time() - t0, 2)

            # 写出结果
            with open(steps_path, "w", encoding="utf-8") as f:
                json.dump(execution_result, f, indent=2, ensure_ascii=False)
            final = execution_result.get("final")
            if isinstance(final, str):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final)

            summary.append({
                "file": file_name,
                "status": "success",
                "duration_sec": duration,
                "output": output_path
            })
            print(f"[Agent] Finished {file_name} in {duration}s")

            # 更新日志
            batch_log[file_name] = {
                "status": "success",
                "start": start_time,
                "end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": duration,
                "output": output_path
            }
            with open(log_path, "w", encoding="utf-8") as lf:
                json.dump(batch_log, lf, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[Agent] Error processing {file_path}: {e}")
            summary.append({
                "file": file_name,
                "status": f"failed: {e}"
            })
            batch_log[file_name] = {
                "status": "failed",
                "error": str(e),
                "start": start_time,
                "end": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(log_path, "w", encoding="utf-8") as lf:
                json.dump(batch_log, lf, indent=2, ensure_ascii=False)

    # 汇总报告
    with open(report_path, "w", encoding="utf-8") as rf:
        for item in summary:
            rf.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("\n[Agent] All tasks completed. Summary report saved to:", report_path)
    print(f"[Agent] Batch log saved to: {log_path}")



if __name__ == "__main__":
    main()
