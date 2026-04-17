import re
import json
import time
import hashlib
import traceback
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import Config
from pipeline import Pipeline


class Memory:
    """Memory module with short-term and long-term stores."""

    def __init__(self):
        self.short_term: Dict[str, Any] = {}
        self.long_term: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def write_short(self, key: str, value: Any):
        with self._lock:
            self.short_term[key] = value

    def read_short(self, key: str, default=None):
        with self._lock:
            return self.short_term.get(key, default)

    def write_long(self, key: str, value: Any):
        with self._lock:
            self.long_term[key] = value

    def read_long(self, key: str, default=None):
        with self._lock:
            return self.long_term.get(key, default)

    def retrieve(self, key: str, default=None):
        """Retrieve a value following short-term then long-term priority order."""
        with self._lock:
            val = self.short_term.get(key)
            if val is not None and val != "":
                return val
            long_val = self.long_term.get(key)
            if long_val is not None and long_val != "":
                return long_val
            return default

    def clear_short_term(self):
        with self._lock:
            self.short_term.clear()

    def checkpoint(self) -> str:
        """Serialize state and compute SHA-256 hash for reproducibility."""
        with self._lock:
            state = {
                "short_term": {k: v for k, v in self.short_term.items() if not k.startswith("_")},
                "long_term": dict(self.long_term),
            }
            serialized = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            return hashlib.sha256(serialized).hexdigest()


class Tool:
    """A single callable tool with symbolic identifier and metadata."""

    def __init__(self, name: str, func: Callable[..., Any], description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ToolRegistry:
    """Centralized tool registry mapping symbolic tool names to callable handlers."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self.tools[name]

    def list_tools(self) -> Dict[str, str]:
        return {name: t.description for name, t in self.tools.items()}

    def verify_all(self, tool_names: List[str]):
        """Verify availability of all tools specified in a plan."""
        for name in tool_names:
            if name not in self.tools:
                raise KeyError(f"Tool '{name}' required by plan but not registered.")


def create_tool_registry(pipeline: Pipeline) -> ToolRegistry:
    """Create the tool registry from a Pipeline instance."""
    registry = ToolRegistry()

    registry.register(Tool(
        "noise_removal",
        func=pipeline.noise_removal,
        description="Remove non-dialogue noise (line-preserving)."
    ))
    registry.register(Tool(
        "content_correction",
        func=pipeline.content_correction,
        description="Correct ASR errors and dialect (line-preserving)."
    ))
    registry.register(Tool(
        "speaker_identification",
        func=pipeline.speaker_identification,
        description="Assign Doctor/Patient/Others labels (line-preserving)."
    ))
    registry.register(Tool(
        "segmentation",
        func=lambda text, strategy="Topic-based": pipeline.apply_segmentation(text, strategy),
        description="Segment conversation by topic/chronology/emotion."
    ))
    registry.register(Tool(
        "segment_topic",
        func=pipeline.segment_by_topic,
        description="Segment conversation by topic transitions."
    ))
    registry.register(Tool(
        "segment_chrono",
        func=pipeline.segment_by_chronology,
        description="Segment conversation by chronological order."
    ))
    registry.register(Tool(
        "segment_emotion",
        func=pipeline.segment_by_emotion,
        description="Segment conversation by emotional transitions."
    ))
    registry.register(Tool(
        "sct_pipeline",
        func=lambda text: pipeline.process_segment(text),
        description="Run the full end-to-end SCT processing pipeline."
    ))

    return registry


class Planner:
    """Planner: interprets task instructions and generates a structured execution plan."""

    VERB_TOOL_MAP = {
        "denoise": "noise_removal", "noise": "noise_removal", "clean": "noise_removal",
        "correct": "content_correction", "correction": "content_correction", "fix": "content_correction", "proofread": "content_correction",
        "speaker": "speaker_identification", "identify": "speaker_identification", "who said": "speaker_identification",
        "topic": "segment_topic", "chronolog": "segment_chrono", "timeline": "segment_chrono", "time order": "segment_chrono",
        "emotion": "segment_emotion", "sentiment": "segment_emotion", "feel": "segment_emotion",
        "segment": "segment_topic",
        "pipeline": "sct_pipeline", "sct": "sct_pipeline", "full pipeline": "sct_pipeline",
    }

    def __init__(self, pipeline: Pipeline, model_name: str = None):
        self.pipeline = pipeline
        self.model_name = model_name or pipeline.cfg.planner_model

    def create_plan(self, task: str, memory: Memory = None) -> List[Dict]:
        """Generate an execution plan from a natural-language task description."""
        try:
            plan = self._llm_plan(task)
        except Exception as e:
            print(f"[Planner] LLM planning failed ({e}), using rule-based fallback.")
            plan = self._fallback_plan(task)

        if memory:
            memory.write_short("plan", plan)

        return plan

    def _llm_plan(self, task: str) -> List[Dict]:
        """Generate plan via LLM call with deterministic configuration."""
        prompt = f"""You are an intelligent planner for a medical text processing agent.
Decompose the user task into a JSON list of tool calls in correct order.

Available tools:
- noise_removal: Remove noise and irrelevant artifacts from transcripts.
- content_correction: Correct grammar, ASR errors, and medical terminology.
- speaker_identification: Label speakers as Doctor, Patient, or Others.
- segment_topic: Segment conversation by topic changes.
- segment_chrono: Segment conversation by chronological order.
- segment_emotion: Segment conversation by emotional transitions.
- sct_pipeline: Execute the full structured clinical transcript pipeline.

Each element must have:
{{"action": "use_tool", "tool": "<tool_name>", "args": ["<input_ref>"], "name": "<step_name>"}}

Use "{{ref:input}}" for the original input, "{{ref:<step_name>}}" for previous step output.
Output a pure JSON array only, no extra text.

User task: {task}"""

        client = self.pipeline.client
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=self.pipeline.cfg.seed,
        )
        content = response.choices[0].message.content
        start = content.find("[")
        json_text = content[start:] if start != -1 else content
        plan = json.loads(json_text)
        if not isinstance(plan, list):
            raise ValueError("Planner returned non-list structure.")
        return plan

    def _fallback_plan(self, task: str) -> List[Dict]:
        """Deterministic rule-based plan generation using VERB_TOOL_MAP."""
        t = task.lower()

        for verb, tool in self.VERB_TOOL_MAP.items():
            if tool == "sct_pipeline" and verb in t:
                return [{"action": "use_tool", "tool": "sct_pipeline", "args": ["{ref:input}"], "name": "sct_pipeline"}]

        pipeline_order = ["noise_removal", "content_correction", "speaker_identification"]
        seg_priority = ["segment_emotion", "segment_chrono", "segment_topic"]

        matched_tools = []
        for verb, tool in self.VERB_TOOL_MAP.items():
            if verb in t and tool not in matched_tools and tool != "sct_pipeline":
                matched_tools.append(tool)

        plan = []
        for tool_name in pipeline_order:
            if tool_name in matched_tools:
                prev = f"{{ref:{plan[-1]['name']}}}" if plan else "{ref:input}"
                plan.append({"action": "use_tool", "tool": tool_name, "args": [prev], "name": tool_name})

        seg_tool = None
        for st in seg_priority:
            if st in matched_tools:
                seg_tool = st
                break

        if seg_tool:
            prev = f"{{ref:{plan[-1]['name']}}}" if plan else "{ref:input}"
            plan.append({"action": "use_tool", "tool": seg_tool, "args": [prev], "name": seg_tool})

        if not plan:
            plan = [
                {"action": "use_tool", "tool": "noise_removal", "args": ["{ref:input}"], "name": "noise_removal"},
                {"action": "use_tool", "tool": "content_correction", "args": ["{ref:noise_removal}"], "name": "content_correction"},
                {"action": "use_tool", "tool": "speaker_identification", "args": ["{ref:content_correction}"], "name": "speaker_identification"},
                {"action": "use_tool", "tool": "segment_topic", "args": ["{ref:speaker_identification}"], "name": "segment_topic"},
            ]

        return plan


class Executor:
    """Executor: invokes tools, validates outputs, and triggers repair or replanning."""

    LINE_PRESERVING_TOOLS = {"noise_removal", "content_correction", "speaker_identification"}
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0

    def __init__(self, pipeline: Pipeline, registry: ToolRegistry, memory: Memory, planner: Planner):
        self.pipeline = pipeline
        self.registry = registry
        self.memory = memory
        self.planner = planner
        self.step_outputs: Dict[str, Any] = {}
        self.step_logs: List[Dict] = []

    def _resolve_arg(self, arg: Any) -> Any:
        """Resolve {ref:name} tokens to actual data from step outputs or Memory."""
        if not isinstance(arg, str):
            return arg
        if arg.startswith("{ref:") and arg.endswith("}"):
            ref = arg[len("{ref:"):-1]
            if ref == "input":
                return self.memory.read_short("input")
            val = self.step_outputs.get(ref)
            if val is not None:
                return val
            return self.memory.read_short(ref)
        return arg

    def _safe_invoke(self, tool: Tool, args: list) -> Any:
        """Safe-execution wrapper with automatic retry on transient errors."""
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return tool.run(*args) if args else tool.run()
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait = self.RETRY_BACKOFF * attempt
                    print(f"[Executor] Retry {attempt}/{self.MAX_RETRIES} for '{tool.name}' after {wait}s: {e}")
                    time.sleep(wait)
        raise RuntimeError(f"Tool '{tool.name}' failed after {self.MAX_RETRIES} retries: {last_error}")

    def _validate_step(self, tool_name: str, input_value: Any, output_value: Any) -> (bool, Optional[str]):
        """Validate tool output against predefined invariants."""
        if isinstance(input_value, str) and isinstance(output_value, str):
            if tool_name in self.LINE_PRESERVING_TOOLS:
                in_count = len(input_value.splitlines())
                out_count = len(output_value.splitlines())
                if in_count != out_count:
                    return False, f"Line count mismatch: input={in_count}, output={out_count}"

        if tool_name.startswith("segment") or tool_name == "segmentation":
            if not output_value or (isinstance(output_value, str) and not output_value.strip()):
                return False, "Segmentation returned empty output"

        if tool_name == "sct_pipeline":
            if not output_value or (isinstance(output_value, str) and not output_value.strip()):
                return False, "SCT pipeline returned empty output"

        return True, None

    def _attempt_repair(self, tool_name: str, input_value: Any) -> Any:
        """Structured repair sequence: chunk-wise, re-invocation, then rule-based fallback."""
        if tool_name in self.LINE_PRESERVING_TOOLS and isinstance(input_value, str):
            chunk_func_map = {
                "noise_removal": self.pipeline._noise_removal_chunk,
                "content_correction": self.pipeline._content_correction_chunk,
                "speaker_identification": self.pipeline._speaker_identification_chunk,
            }
            if tool_name in chunk_func_map:
                try:
                    result = self.pipeline._process_in_chunks(
                        chunk_func_map[tool_name], input_value, f"{tool_name} (repair-stage1)"
                    )
                    ok, _ = self._validate_step(tool_name, input_value, result)
                    if ok:
                        return result
                except Exception as e:
                    print(f"[Executor] Stage 1 repair failed: {e}")

        tool = self.registry.get(tool_name)
        try:
            result = self._safe_invoke(tool, [input_value])
            ok, _ = self._validate_step(tool_name, input_value, result)
            if ok:
                return result
        except Exception as e:
            print(f"[Executor] Stage 2 repair failed: {e}")

        return self._rule_based_fallback(tool_name, input_value)

    def _rule_based_fallback(self, tool_name: str, input_value: Any) -> str:
        """Deterministic rule-based fallback for compliant output generation."""
        if not isinstance(input_value, str):
            return str(input_value) if input_value else ""

        lines = input_value.splitlines()

        if tool_name == "noise_removal":
            noise_patterns = [
                r'[\[【\u201C""].*?(请.*?到|挂号|排队|停车|广播|关闭).*?[】\]\u201D""]',
                r"(电话|环境音|广播)[：:].+",
                r"^(喂|嗯嗯|哦哦|嗯|哈哈)\s*$",
            ]
            result_lines = []
            for line in lines:
                cleaned = line
                for pattern in noise_patterns:
                    cleaned = re.sub(pattern, "", cleaned)
                result_lines.append(cleaned.strip())
            return "\n".join(result_lines)

        elif tool_name == "content_correction":
            return input_value

        elif tool_name == "speaker_identification":
            result_lines = []
            speaker_idx = 0
            for line in lines:
                if not line.strip():
                    result_lines.append(line)
                elif re.match(r"^(Doctor|Patient|Others)\s*[:：]", line):
                    result_lines.append(line)
                else:
                    role = "Doctor" if speaker_idx % 2 == 0 else "Patient"
                    result_lines.append(f"{role}: {line}")
                    speaker_idx += 1
            return "\n".join(result_lines)

        else:
            return f"Topic 1: General Consultation\n{input_value}"

    def _log_step(self, step_name: str, tool_name: str, duration: float,
                  repaired: bool = False, validation_warning: str = None, error: str = None):
        """Record step-level execution metadata."""
        log_entry = {
            "step": step_name,
            "tool": tool_name,
            "timestamp": datetime.now().isoformat(),
            "duration_sec": round(duration, 3),
            "model": self.pipeline.cfg.model_name,
            "temperature": self.pipeline.cfg.temperature,
            "seed": self.pipeline.cfg.seed,
            "memory_checkpoint": self.memory.checkpoint(),
            "repaired": repaired,
        }
        if validation_warning:
            log_entry["validation_warning"] = validation_warning
        if error:
            log_entry["error"] = error
        self.step_logs.append(log_entry)

    def _run_step(self, step: Dict) -> Dict:
        """Execute a single plan step with safe invocation, validation, and repair."""
        if step.get("action") != "use_tool":
            raise ValueError(f"Unsupported action: {step.get('action')}")

        tool_name = step["tool"]
        args = step.get("args", [])
        step_name = step.get("name", f"step_{len(self.step_outputs) + 1}")

        resolved_args = [self._resolve_arg(a) for a in args]

        tool = self.registry.get(tool_name)
        print(f"[Executor] Running '{step_name}' -> {tool_name}")

        t0 = time.time()

        result = self._safe_invoke(tool, resolved_args)

        self.step_outputs[step_name] = result
        self.memory.write_short(step_name, result)

        input_for_validation = resolved_args[0] if resolved_args else None
        ok, msg = self._validate_step(tool_name, input_for_validation, result)

        if not ok:
            print(f"[Executor] Validation failed for '{step_name}': {msg}")

            try:
                repaired = self._attempt_repair(tool_name, input_for_validation)
                ok2, msg2 = self._validate_step(tool_name, input_for_validation, repaired)
                if ok2:
                    print(f"[Executor] Repair succeeded for '{step_name}'")
                    self.step_outputs[step_name] = repaired
                    self.memory.write_short(step_name, repaired)
                    self._log_step(step_name, tool_name, time.time() - t0, repaired=True)
                    return {"step": step_name, "tool": tool_name, "result": repaired, "repaired": True}
                else:
                    print(f"[Executor] Repair still invalid: {msg2}, triggering replanning...")
            except Exception as e:
                print(f"[Executor] Repair failed: {e}, triggering replanning...")

            try:
                repair_task = f"Repair {tool_name} output. Try alternative strategies preserving invariants."
                new_plan = self.planner._fallback_plan(repair_task)
                chain_input = input_for_validation
                for sub_step in new_plan:
                    sub_tool = self.registry.get(sub_step["tool"])
                    chain_input = self._safe_invoke(sub_tool, [chain_input])
                ok3, _ = self._validate_step(tool_name, input_for_validation, chain_input)
                if ok3:
                    self.step_outputs[step_name] = chain_input
                    self.memory.write_short(step_name, chain_input)
                    self._log_step(step_name, tool_name, time.time() - t0, repaired=True)
                    return {"step": step_name, "tool": tool_name, "result": chain_input, "replanned": True}
            except Exception as e:
                print(f"[Executor] Replanning failed: {e}")

            self._log_step(step_name, tool_name, time.time() - t0, validation_warning=msg)
            return {"step": step_name, "tool": tool_name, "result": result, "validation_warning": msg}

        self._log_step(step_name, tool_name, time.time() - t0)
        return {"step": step_name, "tool": tool_name, "result": result}

    def execute_plan(self, plan: List[Dict]) -> Dict:
        """Execute the full plan sequentially with reproducible execution tracing."""
        tool_names = [step["tool"] for step in plan if step.get("action") == "use_tool"]
        self.registry.verify_all(tool_names)

        self.memory.write_short("plan", plan)

        results = []
        for step in plan:
            try:
                out = self._run_step(step)
                results.append(out)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Executor] Error at step '{step.get('name')}': {e}")
                self._log_step(step.get("name", "?"), step.get("tool", "?"), 0, error=str(e))
                results.append({
                    "step": step.get("name"),
                    "tool": step.get("tool"),
                    "error": str(e),
                    "traceback": tb,
                })
                break

        final_output = results[-1].get("result") if results else None
        return {
            "steps": results,
            "final": final_output,
            "step_outputs": dict(self.step_outputs),
            "execution_log": self.step_logs,
            "memory_checkpoint": self.memory.checkpoint(),
        }
