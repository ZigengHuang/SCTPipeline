import os
import re
import json
import time
import glob
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from openai import OpenAI
import PyPDF2
from tqdm import tqdm

from config import Config


class Pipeline:
    """Modular tool layer for medical transcription processing."""

    def __init__(self, config: Config, memory=None):
        self.cfg = config
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
        self.memory = memory
        self.domain_prompts = self._load_domain_prompts()
        self.emotion_data = self._load_emotion_data()

        self._cached_knowledge = None
        self._cached_domain_prompt = None

    def _call_llm(self, messages: list, temperature: float = None) -> str:
        """Send messages to the LLM and return the response text."""
        prompt_text = json.dumps(messages, ensure_ascii=False)
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            seed=self.cfg.seed,
        )
        output = response.choices[0].message.content.strip()

        if self.memory:
            invocation_log = self.memory.read_short("_invocation_log") or []
            invocation_log.append({
                "prompt_hash": prompt_hash,
                "model": self.cfg.model_name,
                "temperature": temperature if temperature is not None else self.cfg.temperature,
                "seed": self.cfg.seed,
                "output_hash": hashlib.sha256(output.encode("utf-8")).hexdigest()[:16],
            })
            self.memory.write_short("_invocation_log", invocation_log)

        return output

    def _add_line_numbers(self, text: str) -> str:
        """Add [ID] tags to each line for tracking alignment."""
        lines = text.splitlines()
        return "\n".join(f"[{i + 1}] {line}" for i, line in enumerate(lines))

    def _remove_line_numbers(self, text: str) -> str:
        """Strip [ID] tags from output lines."""
        lines = text.splitlines()
        return "\n".join(re.sub(r"^\[\d+\]\s?", "", line) for line in lines)

    def _restore_missing_lines(self, original_numbered: str, model_output: str) -> str:
        """Restore missing lines by matching [ID] tags between input and output."""
        orig_lines = original_numbered.splitlines()
        out_lines = model_output.strip().splitlines()

        out_dict = {}
        for line in out_lines:
            match = re.match(r"^\[(\d+)\]", line)
            if match:
                out_dict[int(match.group(1))] = line

        if len(out_dict) == 0:
            restored = []
            for i, orig_line in enumerate(orig_lines):
                if i < len(out_lines):
                    out_line = out_lines[i]
                    if not re.match(r"^\[\d+\]", out_line):
                        orig_match = re.match(r"^(\[\d+\])", orig_line)
                        if orig_match:
                            out_line = f"{orig_match.group(1)} {out_line}"
                    restored.append(out_line)
                else:
                    restored.append(orig_line)
            return "\n".join(restored)

        restored = []
        for i, orig_line in enumerate(orig_lines):
            line_id = i + 1
            if line_id in out_dict:
                restored.append(out_dict[line_id])
            else:
                restored.append(orig_line)
        return "\n".join(restored)

    def _process_in_chunks(self, func, text: str, tool_name: str = "processing") -> str:
        """Split text into chunks, process each with func, and enforce line-count alignment."""
        lines = text.splitlines()
        chunk_size = self.cfg.chunk_size
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        processed_chunks = []

        for chunk in tqdm(chunks, desc=f"  [{tool_name}]", ncols=80, unit="chunk"):
            chunk_text = "\n".join(chunk)
            processed = func(chunk_text)

            out_lines = processed.splitlines()
            if len(out_lines) < len(chunk):
                out_lines += [""] * (len(chunk) - len(out_lines))
            elif len(out_lines) > len(chunk):
                out_lines = out_lines[:len(chunk)]

            processed_chunks.append("\n".join(out_lines))

        return "\n".join(processed_chunks)

    def _get_domain_context(self) -> tuple:
        """Retrieve domain knowledge and prompt following the fixed priority order."""
        knowledge, domain_prompt = "", ""

        if self.memory:
            knowledge = self.memory.retrieve("knowledge", "")
            domain_prompt = self.memory.retrieve("domain_prompt", "")

        if not knowledge:
            if self._cached_knowledge:
                knowledge = self._cached_knowledge
            else:
                try:
                    knowledge = self.load_knowledge()
                    self._cached_knowledge = knowledge
                except Exception:
                    knowledge = ""
        if not domain_prompt:
            if self._cached_domain_prompt:
                domain_prompt = self._cached_domain_prompt
            else:
                try:
                    domain_prompt = self.get_domain_prompt()
                    self._cached_domain_prompt = domain_prompt
                except Exception:
                    domain_prompt = ""

        return knowledge, domain_prompt

    def _noise_removal_chunk(self, text: str) -> str:
        """Remove non-dialogue artifacts from a chunk while preserving line count."""
        numbered = self._add_line_numbers(text)
        knowledge, domain_prompt = self._get_domain_context()

        system_context = ""
        if knowledge or domain_prompt:
            system_context = f"{knowledge}\n\n{domain_prompt}\n\n"

        messages = [
            {"role": "user", "content": f"""{system_context}You are a medical dialogue noise remover.

DELETE these noise patterns from dialogue content:
- Hospital announcements (e.g., "Please go to Room 1 for consultation")
- One-sided phone calls (e.g., "Hello... okay... bye.")
- Environmental sounds and background speech
- Non-medical chatter unrelated to the clinical encounter
- Fragmented, unintelligible speech fragments lacking clinical meaning

KEEP unchanged:
- [ID] numbers at start of each line
- Doctor: / Patient: speaker labels
- All medical dialogue content
- Blank lines

RULES:
1. Output the SAME number of lines as input
2. Each line must start with its original [ID]
3. Only remove noise content, keep everything else
4. NO explanations, output only the cleaned text

Input:
{numbered}

Output starting with [1]:"""}
        ]
        output = self._call_llm(messages, temperature=0.1)
        output = self._restore_missing_lines(numbered, output)
        return self._remove_line_numbers(output)

    def noise_removal(self, text: str) -> str:
        """Remove noise from text (line-preserving contract)."""
        return self._process_in_chunks(self._noise_removal_chunk, text, "Noise Removal")

    def _content_correction_chunk(self, text: str) -> str:
        """Correct ASR errors, dialect, and terminology in a chunk."""
        numbered = self._add_line_numbers(text)

        knowledge, domain_prompt = self._get_domain_context()

        clinical_corr_path = Path(__file__).parent / "knowledge" / "clinical_corrections.txt"
        clinical_corrections = ""
        if clinical_corr_path.exists():
            clinical_corrections = clinical_corr_path.read_text(encoding="utf-8")

        messages = [
            {"role": "system", "content": f"{knowledge}\n\n{domain_prompt}\n\n{clinical_corrections}"},
            {"role": "user", "content": f"""You are a medical transcription corrector with clinical safety awareness.

CORRECT the following issues:
1. Medical terminology errors from ASR misrecognition
2. Dialect words to standard Mandarin
3. Near-homophone errors based on clinical context
4. Only correct individual words; preserve the original spoken expression style

CLINICAL CRITICAL ERROR CHECK (high priority):
5. Negation errors: verify negation words are correctly placed
6. Medication/procedure names: verify drug names and medical procedure terms match context
7. Numeric/dosage values: verify numbers are clinically plausible
8. Temporal expressions: verify time units are clinically appropriate
9. Causal logic: verify cause-effect relationships in medical statements are preserved

RULES:
1. Output the SAME number of lines as input
2. Preserve each [ID] tag exactly
3. Do NOT merge, split, or reorder lines
4. Keep speaker labels unchanged
5. NO explanations, output only the corrected text

Input:
{numbered}

Output starting with [1]:"""}
        ]
        output = self._call_llm(messages, temperature=0.1)
        output = self._restore_missing_lines(numbered, output)
        return self._remove_line_numbers(output)

    def content_correction(self, text: str) -> str:
        """Correct content errors (line-preserving contract)."""
        return self._process_in_chunks(self._content_correction_chunk, text, "Content Correction")

    def _speaker_identification_chunk(self, text: str) -> str:
        """Assign Doctor/Patient/Others labels to each line in a chunk."""
        numbered = self._add_line_numbers(text)

        knowledge, domain_prompt = self._get_domain_context()
        domain_section = f"{domain_prompt}\n\n" if domain_prompt else ""

        messages = [
            {"role": "system", "content": f"""{domain_section}You are a speaker labeler for medical dialogues.
Re-label the speaker for each line based on content and context.

Valid labels:
- Doctor: gives medical advice, asks about symptoms, explains procedures
- Patient: describes symptoms, asks questions, expresses concerns
- Others: family members, assistants, or background staff

RULES:
- Output the SAME number of lines as input
- Preserve every [ID] exactly as input
- Do NOT merge, split, delete, or reorder lines
- Keep the original content after the label unchanged
- Output only relabeled lines, nothing else"""},
            {"role": "user", "content": numbered}
        ]
        output = self._call_llm(messages, temperature=0.1)
        output = self._restore_missing_lines(numbered, output)
        return self._remove_line_numbers(output)

    def speaker_identification(self, text: str) -> str:
        """Identify speakers (line-preserving contract)."""
        return self._process_in_chunks(self._speaker_identification_chunk, text, "Speaker Identification")

    def segment_by_topic(self, text: str) -> str:
        """Segment dialogue by topic transitions."""
        cleaned = self._remove_timestamps(text)
        topic_templates = self._get_topic_templates()
        system_prompt = f"""Domain: {self.cfg.domain_name}
Topic list: {topic_templates}

Identify turning points in topics and segment the medical conversation accordingly.
Return results in the format:
Topic 1: [Topic name]
[Conversation lines]

Topic 2: [Topic name]
[Conversation lines]"""

        output = self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cleaned}
        ], temperature=0.3)
        return self._validate_topic_format(output)

    def segment_by_chronology(self, text: str) -> str:
        """Segment dialogue by temporal order."""
        dialogues = self._parse_timestamped_lines(text)
        segments, current, prev_time = [], [], None

        for turn in dialogues:
            if prev_time and (turn["seconds"] - prev_time) > 60:
                if len(current) >= 3:
                    segments.append(current)
                    current = []
            current.append(turn)
            prev_time = turn["seconds"]

        if current:
            segments.append(current)
        return self._format_chrono_segments(segments)

    def segment_by_emotion(self, text: str) -> str:
        """Segment dialogue by emotional state transitions."""
        cleaned = self._remove_timestamps(text)
        enhanced = self._enhance_emotion_tags(cleaned)
        output = self._call_llm([
            {"role": "system", "content": (
                "Segment the conversation based on changes in the patient's emotional state. "
                "Ensure completeness of each segment.\n\n" + enhanced
            )},
            {"role": "user", "content": cleaned}
        ], temperature=0.5)
        return output

    def apply_segmentation(self, text: str, strategy: str = "Topic-based") -> str:
        """Apply the specified segmentation strategy."""
        strategies = {
            "Topic-based": self.segment_by_topic,
            "Chronological": self.segment_by_chronology,
            "Emotion-based": self.segment_by_emotion,
        }
        if strategy not in strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Choose from {list(strategies.keys())}")
        return strategies[strategy](text)

    def _load_domain_prompts(self) -> dict:
        """Load domain-specific prompt templates from JSON."""
        prompt_path = self.cfg.project_root / "prompts" / "domain_specific.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Domain prompts not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_emotion_data(self) -> dict:
        """Load emotion feature definitions."""
        path = self.cfg.project_root / "prompts" / "emotion_features.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "Emotions" in raw and isinstance(raw["Emotions"], list):
            return {
                item["Type"]: {"Features": item.get("Features", ""), "Examples": item.get("Examples", [])}
                for item in raw["Emotions"]
                if "Type" in item
            }

        if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
            return raw

        return {}

    def get_domain_prompt(self) -> str:
        """Retrieve and format the domain-specific prompt for the current domain."""
        domain_data = self.domain_prompts.get(self.cfg.domain_index)
        if not domain_data:
            domain_data = self.domain_prompts.get(self.cfg.domain_name)
        if not domain_data:
            return (
                f"# Domain Description\n"
                f"General medical conversation processing for {self.cfg.domain_name}.\n\n"
                f"# Identification Topics\n"
                f"- Symptom inquiry\n- Medication guidance\n- Follow-up arrangement"
            )

        parts = []
        if desc := domain_data.get("description"):
            parts.append(f"# Domain Description\n{desc.strip()}")
        if topics := domain_data.get("topics"):
            parts.append("\n# Identification Topics")
            for topic, details in topics.items():
                clean_topic = topic.replace(":", "").strip()
                summary = details.split("\n")[0].strip()
                parts.append(f"- {clean_topic}: {summary}")
        return "\n".join(parts)

    def load_knowledge(self) -> str:
        """Load all domain knowledge files (txt + pdf) from the knowledge base."""
        knowledge_parts = []
        files = sorted(
            glob.glob(f"{self.cfg.knowledge_base}/*.txt")
            + glob.glob(f"{self.cfg.knowledge_base}/*.pdf")
        )
        for file_path in files:
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    knowledge_parts.append(f.read())
            elif file_path.endswith(".pdf"):
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(
                        page.extract_text() for page in reader.pages if page.extract_text()
                    )
                    knowledge_parts.append(text)

        return (
            f"Medical and dialectal knowledge for {self.cfg.domain_name.upper()} domain. "
            "Use this information for subsequent structured processing.\n"
            + "\n".join(knowledge_parts)
        )

    def presegment_ruct(self, text: str) -> List[str]:
        """Pre-segment RUCT into batches for manageable context length."""
        if not text.strip():
            return []

        segments = []
        current = []
        char_count = 0

        for paragraph in text.split("\n\n"):
            p_len = len(paragraph)
            if char_count + p_len > self.cfg.segment_max_chars and char_count >= self.cfg.segment_min_chars:
                segments.append("\n\n".join(current))
                current = []
                char_count = 0
            current.append(paragraph)
            char_count += p_len

        if current:
            segments.append("\n\n".join(current))
        return segments

    def process_segment(self, segment: str, segmentation_strategy: str = "Topic-based") -> str:
        """Process a single RUCT segment through all four tools sequentially."""
        cleaned = self.noise_removal(segment)
        corrected = self.content_correction(cleaned)
        identified = self.speaker_identification(corrected)
        segmented = self.apply_segmentation(identified, segmentation_strategy)
        return segmented

    def generate_output(self, processed_segments: List[str]) -> str:
        """Merge processed segments into a unified SCT."""
        return "\n\n".join(processed_segments)

    def _remove_timestamps(self, text: str) -> str:
        """Remove timestamp patterns from line starts."""
        lines = text.split("\n")
        cleaned = [re.sub(r"^\d+:\d+(?::\d+)?\s*", "", line.strip()) for line in lines]
        return "\n".join(line for line in cleaned if line)

    def _get_topic_templates(self) -> str:
        """Return topic templates for the current domain."""
        domain_data = self.domain_prompts.get(self.cfg.domain_index) or self.domain_prompts.get(self.cfg.domain_name)
        if domain_data and "topics" in domain_data:
            return ", ".join(t.replace(":", "").strip() for t in domain_data["topics"].keys())

        name_lower = self.cfg.domain_name.lower()
        if "surg" in name_lower:
            return "Preoperative preparation, Surgical procedure, Postoperative care"
        elif "hospital" in name_lower:
            return "Admission guidance, Discharge management, Medication precautions"
        elif "checkup" in name_lower:
            return "Examination preparation, Health counseling, Result interpretation"
        return "Symptom inquiry, Medication guidance, Follow-up arrangement"

    def _validate_topic_format(self, text: str) -> str:
        """Ensure topic-based segmentation output is properly formatted."""
        pattern = r"(Topic \d+: .+?)\n((?:.+\n)+?)(?=Topic \d+:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n\n".join(f"{title}\n{content.strip()}" for title, content in matches)
        return text

    def _parse_timestamped_lines(self, text: str) -> List[Dict]:
        """Parse conversation lines with timestamps for chronological segmentation."""
        dialogues = []
        for line in text.split("\n"):
            match = re.match(r"^(\S+)\s+(\d+:\d+:\d+)\s+(.*)", line)
            if match:
                role, ts, content = match.groups()
                dialogues.append({"role": role, "seconds": self._time_to_seconds(ts), "content": content})
        return dialogues

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert HH:MM:SS or MM:SS to seconds."""
        parts = list(map(int, time_str.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return 0

    def _format_chrono_segments(self, segments: List[List[Dict]]) -> str:
        """Format chronological segments into readable output."""
        output = []
        for idx, seg in enumerate(segments, 1):
            output.append(f"Segment {idx}:")
            output.extend(f"  {turn['role']}: {turn['content']}" for turn in seg)
            output.append("")
        return "\n".join(output)

    def _enhance_emotion_tags(self, text: str) -> str:
        """Enhance emotional expressions using emotion feature definitions."""
        if not self.emotion_data:
            return text
        emotion_prompt = "\n".join(
            f"{emo}: {info['Features']} Example: {', '.join(info['Examples'])}"
            for emo, info in self.emotion_data.items()
        )
        return self._call_llm([
            {"role": "system", "content": (
                "Based on the six basic emotions, enhance the patient's emotional expression. "
                "Modify punctuation, add interjections, or include brief annotations. "
                f"Do not alter semantic content.\n\n{emotion_prompt}"
            )},
            {"role": "user", "content": text}
        ], temperature=0.5)
