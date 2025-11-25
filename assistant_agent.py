"""
assistant_agent.py
Usage:
    python assistant_agent.py --input path/to/input.ruct --output path/to/output_dir --task "your tasks"
    python assistant_agent.py --input /Users/qinhao/PUMC_Project/Protocol/agent/GPT_Agent/ruct.txt --output /Users/qinhao/PUMC_Project/Protocol/agent/GPT_Agent --task "Please denoise and correct this clinical note"
    python assistant_agent.py --input /Users/qinhao/PUMC_Project/Protocol/agent/GPT_Agent/ruct.txt --output /Users/qinhao/PUMC_Project/Protocol/agent/GPT_Agent --task "Please denoise, correct and identify speaker this clinical note"
    python assistant_agent.py --input /Users/qinhao/PUMC_Project/Protocol/pre_data/0424内窥镜.txt --output /Users/qinhao/PUMC_Project/Protocol/agent/GPT_Agent --task "Please denoise, correct and identify speaker this clinical note"
    python assistant_agent.py --input /Users/qinhao/PUMC_Project/Protocol/testdata --output /Users/qinhao/PUMC_Project/Protocol/agent/QWEN_Agent --task "Please denoise, correct and identify speaker this clinical note"
"""
import argparse
import traceback
import sys
import json
import os
from tqdm import tqdm
from datetime import datetime
from typing import Any, Callable, Dict, List

# Import the existing pipeline (must be in same dir or installable module)
from medical_transcription_pipeline import Config, MedicalTranscriptionPipeline

# ---------- Memory module ----------
class Memory:
    """
    A tiny memory store for short-term (per-task) and optional long-term storage.
    Here it's just an in-memory dictionary. You can persist to disk if needed.
    """
    def __init__(self):
        self.short_term = {}
        self.long_term = {}

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
    Planner tries to use the pipeline's OpenAI client to call model 'gpt-5-nano' to convert a
    natural language task description into a JSON plan: a list of steps, each step is:
      {"action": "use_tool", "tool": "denoise", "args": ["<text or ref>"], "name": "step1"}
    If the model call fails (e.g. model not available), fallback to simple rule-based planner.
    """
    def __init__(self, pipeline: MedicalTranscriptionPipeline, model_name: str = "deepseek-v3", temperature: float = 0.0):
        self.pipeline = pipeline
        self.model_name = model_name
        self.temperature = temperature

    def _call_model_for_plan(self, task: str) -> List[Dict]:
        # Template prompt instructing output as pure JSON list
        prompt = f"""
You are a planner that outputs a JSON array of steps to achieve the user's task.
The available tools are: denoise, correction, speaker_identify, segmentation, preprocess, sct_pipeline.
Each step must be an object with fields:
 - action: must be "use_tool"
 - tool: one of the available tools
 - args: list of arguments. If referencing an earlier step's output, use the special token: {{ref:stepN}}
 - name: unique step name like "step1"

User task: {task}

Return ONLY parseable JSON array. Example:
[
  {{"action":"use_tool","tool":"denoise","args":["{{ref:input}}"],"name":"step1"}},
  {{"action":"use_tool","tool":"correction","args":["{{ref:step1}}"],"name":"step2"}}
]
"""
        # Use the pipeline's client if available
        try:
            client = getattr(self.pipeline, "client", None)
            if client is None:
                raise RuntimeError("No client attached to pipeline for planning.")
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role":"user","content": prompt}],
                temperature=self.temperature
            )
            content = response.choices[0].message.content
            # Some models sometimes wrap JSON in text; try to find first '['
            start = content.find('[')
            if start != -1:
                json_text = content[start:]
            else:
                json_text = content
            plan = json.loads(json_text)
            return plan
        except Exception as e:
            # Bubble up trace for debugging then fallback
            print("Planner model call failed or returned non-JSON. Falling back to simple planner.")
            print("Model exception:", str(e))
            # print(traceback.format_exc())
            return self._fallback_plan(task)

    def _fallback_plan(self, task: str) -> List[Dict]:
        """A small deterministic planner mapping common verbs to tools."""
        t = task.lower()
        plan = []
        # default: input reference
        if "run pipeline" in t or "whole pipeline" in t or "run the pipeline" in t:
            plan = [{"action":"use_tool","tool":"sct_pipeline","args":["{ref:input}"],"name":"run_all"}]
        else:
            # sequence: preprocess -> denoise -> correction -> speaker -> segmentation -> sct_pipeline/assemble
            if "preprocess" in t or "pre-seg" in t or "segment" in t and "preprocess" not in t:
                plan.append({"action":"use_tool","tool":"preprocess","args":["{ref:input}"],"name":"preprocess"})
            if "denoise" in t or "noise" in t or "clean" in t:
                plan.append({"action":"use_tool","tool":"denoise","args":["{ref:input}"],"name":"denoise"})
            if "correct" in t or "correction" in t or "fix" in t:
                plan.append({"action":"use_tool","tool":"correction","args":["{ref:denoise}" if any(p["tool"]=="denoise" for p in plan) else "{ref:input}"],"name":"correction"})
            if "speaker" in t or "identify speaker" in t or "speaker id" in t:
                plan.append({"action":"use_tool","tool":"speaker_identify","args":["{ref:correction}" if any(p["tool"]=="correction" for p in plan) else "{ref:input}"],"name":"speaker_identify"})
            if "segment" in t or "segmentation" in t:
                plan.append({"action":"use_tool","tool":"segmentation","args":["{ref:speaker_identify}" if any(p["tool"]=="speaker_identify" for p in plan) else "{ref:input}"],"name":"segmentation"})
            # If still empty, do denoise + correction as a reasonable default
            if not plan:
                plan = [
                    {"action":"use_tool","tool":"denoise","args":["{ref:input}"],"name":"denoise"},
                    {"action":"use_tool","tool":"correction","args":["{ref:denoise}"],"name":"correction"}
                ]
        return plan

    def create_plan(self, task: str) -> List[Dict]:
        # Always wrap input reference
        plan = self._call_model_for_plan(task)
        # validate plan shape
        if not isinstance(plan, list):
            raise ValueError("Planner returned unexpected structure.")
        return plan

# ---------- Agent Executor ----------
class Agent:
    def __init__(self, pipeline: MedicalTranscriptionPipeline, tool_registry: ToolRegistry, memory: Memory):
        self.pipeline = pipeline
        self.tools = tool_registry
        self.memory = memory
        self.step_outputs = {}  # store outputs by step name

    def _resolve_arg(self, arg: str):
        """If arg is a reference token like {ref:step1} or {ref:input}, resolve it."""
        if not isinstance(arg, str):
            return arg
        if arg.startswith("{ref:") and arg.endswith("}"):
            ref = arg[len("{ref:"):-1]
            # first check step_outputs then memory short term then 'input'
            if ref == "input":
                return self.memory.read_short("input")
            return self.step_outputs.get(ref) or self.memory.read_short(ref)
        return arg

    def _run_step(self, step: Dict) -> Dict:
        if step.get("action") != "use_tool":
            raise ValueError(f"Unsupported action: {step.get('action')}")
        tool_name = step.get("tool")
        args = step.get("args", [])
        step_name = step.get("name") or f"step_{len(self.step_outputs)+1}"

        # Resolve args (may reference previous outputs)
        resolved_args = [self._resolve_arg(a) for a in args]

        tool = self.tools.get(tool_name)
        print(f"Executing step '{step_name}' -> tool '{tool_name}' with args: {[type(a).__name__ for a in resolved_args]}")
        try:
            result = tool.run(*resolved_args)
        except TypeError:
            # try passing as single text if list contains only 1 string
            if len(resolved_args) == 1:
                result = tool.run(resolved_args[0])
            else:
                raise
        except Exception as e:
            print(f"Error running tool {tool_name}: {e}")
            raise

        # save result
        self.step_outputs[step_name] = result
        return {"step": step_name, "tool": tool_name, "result": result}

    def execute_plan(self, plan: List[Dict]) -> Dict:
        results = []
        for step in plan:
            try:
                out = self._run_step(step)
                results.append(out)
            except Exception as e:
                # On error, include traceback and stop execution
                tb = traceback.format_exc()
                results.append({"step": step.get("name"), "tool": step.get("tool"), "error": str(e), "traceback": tb})
                break
        # final return: all step outputs and a 'final' best guess
        final_output = results[-1].get("result") if results else None
        return {"steps": results, "final": final_output, "all_step_outputs": self.step_outputs}

# ---------- Tool wrappers mapping to pipeline's methods ----------
def create_tools_from_pipeline(pipeline: MedicalTranscriptionPipeline) -> ToolRegistry:
    registry = ToolRegistry()

    # Preprocess -> returns list of segments (call pipeline.preprocess_ruct)
    registry.register(Tool(
        "preprocess",
        func=lambda text_or_path: pipeline.preprocess_ruct() if text_or_path == "{auto}" else pipeline.preprocess_ruct() ,
        description="Pre-segment input RUCT into smaller segments. When called in agent, input is read from pipeline.cfg.input_ruct."
    ))

    # denoise -> wrap _remove_noise which expects text
    registry.register(Tool(
        "denoise",
        func=lambda text: pipeline._remove_noise(text),
        description="Remove noise from a text segment."
    ))

    # correction -> wrap _correct_content
    registry.register(Tool(
        "correction",
        func=lambda text: pipeline._correct_content(text),
        description="Apply domain-aware content correction."
    ))

    # speaker_identify
    registry.register(Tool(
        "speaker_identify",
        func=lambda text: pipeline._identify_speakers(text),
        description="Identify speakers line-by-line and label as Doctor/Patient/Others."
    ))

    # segmentation wrapper -> expects text & strategy optional. For agent, use default strategy 'Topic-based'
    registry.register(Tool(
        "segmentation",
        func=lambda text, strategy="Topic-based": pipeline.apply_segmentation(text, strategy),
        description="Segment processed text; default strategy: Topic-based. Args: (text, [strategy])"
    ))

    # sct_pipeline -> run end-to-end pipeline on input file (wrap run_pipeline can be interactive -> we instead call process_segment series for automation)
    def sct_pipeline_auto(input_text):
        """
        A safe wrapper that:
         - If input_text is a str that's long, run preprocess->process_segment for each segment and assemble result.
         - If input_text is a list of segments, process each.
        """
        # If user passed raw text, temporarily set as pipeline input by creating a temp file is more complex.
        # Simpler: if input is list -> treat as segments; if string -> treat as a single segment.
        if isinstance(input_text, list):
            segments = input_text
        else:
            segments = [input_text]

        processed = []
        for seg in segments:
            processed.append(pipeline.process_segment(seg))
        return pipeline.generate_output(processed)

    registry.register(Tool(
        "sct_pipeline",
        func=sct_pipeline_auto,
        description="Run an automated pipeline on provided segments or text; returns assembled SCT output."
    ))

    return registry

# ---------- CLI Entrypoint ----------
def main():
    parser = argparse.ArgumentParser(description="Agent wrapper for medical transcription pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input file or folder path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--task", "-t", required=True, help="Task description (e.g. 'Please denoise, correct and identify speaker this clinical note')")
    parser.add_argument("--planner-model", default="gpt-5-nano", help="Model name for planner (default gpt-5-nano)")
    parser.add_argument("--resume", action="store_true", help="Resume mode: skip already processed files and use previous log")
    parser.add_argument("--domain", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Medical domain index (1-5). Default=1 (Health checkups)")
    args = parser.parse_args()

    import sys
    import json
    import os
    import time
    from tqdm import tqdm
    from datetime import datetime, timedelta

    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, "batch_log.json")
    report_path = os.path.join(args.output, "summary_report.txt")

    # 加载历史日志以便断点续跑
    processed_log = {}
    if args.resume and os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                processed_log = json.load(f)
            print(f"🔄 Resume mode enabled. Loaded {len(processed_log)} previous entries from log.")
        except Exception:
            print("⚠️ Failed to load previous log, starting fresh.")
            processed_log = {}

    # 文件夹输入：批量模式
    if os.path.isdir(args.input):
        all_txt_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".txt")]
        total_files = len(all_txt_files)
        print(f"📂 Detected folder input: {total_files} text files found in {args.input}.")
        print(f"🩺 Using medical domain = {args.domain}\n")

        success, skipped, failed = 0, 0, 0
        start_all = time.time()

        # 覆盖 Config.select_medical_domain，跳过交互选择
        def _auto_select_domain(self):
            self.medical_domain, self.domain_key, self.knowledge_base = self.domain_mapping[args.domain]
            return self.medical_domain, self.domain_key, self.knowledge_base

        Config.select_medical_domain = _auto_select_domain

        for file_path in tqdm(all_txt_files, desc="Processing files", unit="file"):
            file_name = os.path.basename(file_path)
            output_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(args.output, f"{output_name}.txt")

            # 跳过逻辑（断点续跑模式）
            if args.resume and processed_log.get(file_name, {}).get("status") == "success":
                tqdm.write(f"⏭️  Skipping {file_name} (marked successful in log)")
                skipped += 1
                continue
            if os.path.exists(output_path):
                tqdm.write(f"⏭️  Skipping {file_name} (output already exists)")
                skipped += 1
                continue

            # 初始化 Config，禁用交互
            original_argv = sys.argv.copy()
            try:
                sys.argv = [original_argv[0], "-i", file_path, "-o", args.output]
                config = Config()
                config.medical_domain, config.domain_key, config.knowledge_base = config.domain_mapping[args.domain]
                config.knowledge_base = os.path.abspath(config.knowledge_base)
            finally:
                sys.argv = original_argv

            start_time = time.time()
            status_record = {
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "domain": args.domain,
                "file": file_name
            }

            try:
                # 初始化流水线与 agent
                pipeline = MedicalTranscriptionPipeline(config)
                registry = create_tools_from_pipeline(pipeline)
                memory = Memory()

                # 读取文本内容
                with open(file_path, "r", encoding="utf-8") as f:
                    input_text = f.read()
                memory.write_short("input", input_text)

                # 规划 + 执行
                planner = Planner(pipeline, model_name=args.planner_model)
                agent = Agent(pipeline, registry, memory)

                plan = planner.create_plan(args.task)
                execution_result = agent.execute_plan(plan)

                # 保存输出
                final = execution_result.get("final", "")
                with open(output_path, "w", encoding="utf-8") as f:
                    if isinstance(final, str):
                        f.write(final)
                    else:
                        f.write(json.dumps(final, indent=2, ensure_ascii=False))

                success += 1
                tqdm.write(f"✅ Finished {file_name}")
                status_record["status"] = "success"

            except Exception as e:
                failed += 1
                tqdm.write(f"❌ Error processing {file_name}: {e}")
                status_record["status"] = "failed"
                status_record["error"] = str(e)

            finally:
                end_time = time.time()
                status_record["end_time"] = datetime.now().isoformat()
                status_record["duration_seconds"] = round(end_time - start_time, 2)
                processed_log[file_name] = status_record

                # 实时写入日志
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(processed_log, f, indent=2, ensure_ascii=False)

        # 汇总结果
        total_duration = time.time() - start_all
        total_success_time = sum(v.get("duration_seconds", 0) for v in processed_log.values() if v.get("status") == "success")
        avg_time = total_success_time / success if success > 0 else 0

        total_time_str = str(timedelta(seconds=int(total_duration)))

        print("\n🎯 Batch processing complete.")
        print(f" -  Success: {success}")
        print(f" -  Skipped: {skipped}")
        print(f" -  Failed: {failed}")
        print(f" -  Avg duration: {avg_time:.2f}s per file")
        print(f" -  Total time: {total_time_str}")
        print(f" -  Outputs saved to: {args.output}")
        print(f" -  Log saved to: {log_path}")

        # 生成 summary_report.txt
        with open(report_path, "w", encoding="utf-8") as report:
            report.write("=== Batch Processing Summary ===\n")
            report.write(f"Start time: {datetime.fromtimestamp(start_all)}\n")
            report.write(f"End time: {datetime.now()}\n")
            report.write(f"Total files: {total_files}\n")
            report.write(f"Success: {success}\n")
            report.write(f"Skipped: {skipped}\n")
            report.write(f"Failed: {failed}\n")
            report.write(f"Average duration per file: {avg_time:.2f}s\n")
            report.write(f"Total elapsed time: {total_time_str}\n\n")

            report.write("=== Per-file Details ===\n")
            for name, rec in processed_log.items():
                dur = rec.get("duration_seconds", "N/A")
                st = rec.get("status", "?")
                report.write(f"{name:<30} {st:<10} {dur}s\n")

        print(f"📄 Summary report generated: {report_path}")

    # 单文件模式（保持原逻辑）
    else:
        print(f"Processing single file: {args.input}")
        original_argv = sys.argv.copy()
        try:
            sys.argv = [original_argv[0], "-i", args.input, "-o", args.output]
            config = Config()
            config.medical_domain, config.domain_key, config.knowledge_base = config.domain_mapping[args.domain]
            config.knowledge_base = os.path.abspath(config.knowledge_base)
        finally:
            sys.argv = original_argv

        pipeline = MedicalTranscriptionPipeline(config)
        registry = create_tools_from_pipeline(pipeline)
        memory = Memory()

        with open(args.input, "r", encoding="utf-8") as f:
            input_text = f.read()
        memory.write_short("input", input_text)

        planner = Planner(pipeline, model_name=args.planner_model)
        agent = Agent(pipeline, registry, memory)

        print("Creating plan for task:", args.task)
        plan = planner.create_plan(args.task)

        print("Executing plan...")
        start_time = time.time()
        execution_result = agent.execute_plan(plan)
        duration = round(time.time() - start_time, 2)

        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "agent_execution_steps.json"), "w", encoding="utf-8") as f:
            json.dump(execution_result, f, indent=2, ensure_ascii=False)

        final = execution_result.get("final")
        if isinstance(final, str):
            with open(os.path.join(out_dir, "agent_final_output.txt"), "w", encoding="utf-8") as f:
                f.write(final)

        print("\nExecution finished. Results saved to:", out_dir)
        print("Summary:")
        print(f" - Duration: {duration}s")
        print(" - Steps:", len(execution_result.get("steps", [])))
        if final:
            print(" - Final output saved as agent_final_output.txt")



if __name__ == "__main__":
    main()
