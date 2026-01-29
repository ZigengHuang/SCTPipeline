import os
import argparse
import re
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import sentencepiece as spm
from openai import OpenAI
import PyPDF2
import json
import tempfile

# ====================
# Set the parameters
# ====================
class Config:
    def __init__(self):
        # API
        self.api_key = "api_key"
        self.model_name = "model_name" #deepseek-v3-2-exp/gpt-4o/gpt-5-mini/gemini-2.5-flash/gemini-2.5-pro/gemini-2.5-flash-nothinking/grok-4-fast/gpt-4o-mini/gpt-5-mini-ca/qwen3-235b-a22b-instruct-2507
        self.temperature = 0.1

        # Parameters for argparse
        parser = argparse.ArgumentParser(
            description="Preprocessing Real-World Medical Conversation Transcriptions for LLM Development",
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            "-i", "--input",
            required=True,
            help="Input RUCT file path"
        )
        parser.add_argument(
            "-o", "--output",
            required=True,
            help="Output SCT file path"
        )
        parser.add_argument("--domain", type=int, default=1, help="Medical domain number (1-5)")
        args = parser.parse_args()

        # File path
        self.input_ruct = os.path.abspath(args.input)
        self.output_dir = os.path.abspath(args.output)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Output directory '{self.output_dir}' did not exist, created it.")

        print(f"Debug: Input path resolved to {self.input_ruct}")
        print(f"Debug: Output directory resolved to {self.output_dir}")

        if not os.path.exists(self.input_ruct):
            raise FileNotFoundError(f"Input file not found: {self.input_ruct}")

        # Mapping of departments and medical knowledge base paths
        self.domain_mapping = {
            1: ("Health checkups", "Health checkups", "./SCTPipeline/medical_knowledge_backup/Health checkups medical_knowledge"),
            2: ("General outpatient visits", "General outpatient visits", "./SCTPipeline/medical_knowledge_backup/Routine outpatient guidance medical_knowledge"),
            3: ("Surgical procedures", "Surgical procedures", "./SCTPipeline/medical_knowledge_backup/Surgery medical_knowledge"),
            4: ("Hospitalization management", "Hospitalization management", "./SCTPipeline/medical_knowledge_backup/Hospitalization guidance medical_knowledge"),
            5: ("Customized domain-specific features", "Customized domain-specific features", "./SCTPipeline/medical_knowledge_backup/option_5_General medical_knowledge")
        }

        # Select the domain
        self.medical_domain, self.domain_key, self.knowledge_base = self.select_medical_domain()
        self.knowledge_base = os.path.abspath(self.knowledge_base)

        # Make sure the path exists
        if not os.path.exists(self.knowledge_base):
            raise FileNotFoundError(f"The knowledge base path does not exist：{self.knowledge_base}")
        
        # Find SCT.txt
        self.sct_file = (Path(self.knowledge_base) / "SCT.txt").resolve()
        print(f"Debug: Resolved SCT.txt path: {self.sct_file}, Exists: {self.sct_file.exists()}")

        if not self.sct_file.exists():
            print(f"Debug: Files in {self.knowledge_base}: {os.listdir(self.knowledge_base)}")
            raise FileNotFoundError(f"SCT.txt not found in {self.knowledge_base}.")
        
        # Processing parameters
        self.vocab_size = 800
        self.similarity_threshold = 0.9

    def select_medical_domain(self):
        """Select medical domain"""
        print("\n" + "="*30)
        print("Please choose the medical domain：")
        for num, (domain, key, path) in self.domain_mapping.items():
            print(f"{num}. {domain}")

        while True:
            choice = input("If you choose option 5, please replace the SCT file yourself\nPlease enter the option number (1-5，Default 1): ").strip()
            if not choice:
                return self.domain_mapping[1]
            
            try:
                choice_num = int(choice)
                if choice_num == 5:
                    print("Customized domain-specific features selected. Loading general knowledge base.")
                if choice_num in self.domain_mapping:
                    self.domain_key = choice_num
                    return self.domain_mapping[choice_num]
                print("Please enter a valid number between 1-5.")
            except ValueError:
                print("Invalid input, please re-enter.")

        
    def _generate_custom_domain(self, sct_path: str) -> str:
        """Generate custom domain-specific features based on the provided SCT file"""
        with open(sct_path, 'r', encoding='utf-8') as f:
            sct_content = f.read()
        
        pipeline = MedicalTranscriptionPipeline(self)
        return pipeline.create_custom_domain_agent(sct_content)
    
   
# ====================
# Procedures
# ====================
class MedicalTranscriptionPipeline:
    def __init__(self, config: Config):
        self.cfg = config
        self.client = OpenAI(
            api_key="sk-VsjuaibkxCYRKJB3zS9LiClI1RAkyKZrTKmIkruDAZkH57D1",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        # Initialization
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.emotion_data = self._load_emotion_data("Features of patient emotions.json")

        self.segmentation_strategies = {
            'Topic-based': self._segment_by_topic,
            'Chronological': self._segment_by_chronological,
            'Emotion-based': self._segment_by_emotion
        }

        self._interim_results = []
        self.domain_prompts = self._load_domain_prompts()

        self.build_wordlist()
        self.emotion_data = None

    # ====================
    # Phase 1-2: RUCT for batches pre-segmentation (Steps 4-6)
    # ====================
    def _segment_text(self, text: str) -> List[str]:
        """Pre-segmentation"""
        segments = []
        current_segment = []
        char_count = 0
        
        for paragraph in text.split('\n\n'):
            p_len = len(paragraph)
            if char_count + p_len > 3000 and char_count >= 2000:
                segments.append('\n\n'.join(current_segment))
                current_segment = []
                char_count = 0
            current_segment.append(paragraph)
            char_count += p_len
        
        if current_segment:
            segments.append('\n\n'.join(current_segment))
        return segments

    def preprocess_ruct(self) -> List[str]:
        """Main for RUCT pre-segmentation"""
        with open(self.cfg.input_ruct, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self._segment_text(text) 

    # ====================
    # Phase 2-1: Domain-Specific Features (Step 7,predefined prompt)
    # ====================
    def _load_domain_prompts(self) -> dict:
        """Loading the domain specific prompts json files"""
        prompt_path = Path(__file__).parent / "Domain Specific Prompts.json"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 不强转，直接返回
            return data
        except FileNotFoundError:
            raise RuntimeError(f"Cannot find the json file：{prompt_path}")
        except json.JSONDecodeError:
            raise RuntimeError("The file format is incorrect. Please check the JSON format.")

    def _format_domain_prompt(self, domain_data: dict) -> str:
        """Convert the JSON to the prompt string"""
        prompt = []
        
        if desc := domain_data.get("description"):
            prompt.append(f"# Domain desciption\n{desc.strip()}")
        
        if topics := domain_data.get("topics"):
            prompt.append("\n# Identification topics")
            for topic, details in topics.items():
                clean_topic = topic.replace(":", "").strip()
                summary = details.split('\n')[0].strip()
                prompt.append(f"- {clean_topic}：{summary}")
        
        return "\n".join(prompt)

    def get_domain_prompt(self) -> str:
        """Get domain prompt"""
        if self.cfg.medical_domain.startswith("Customized domain-specific features"):
            return self.cfg.medical_domain
        else:
            # 优先尝试 domain_key (int)
            if isinstance(self.cfg.domain_key, int):
                if domain_data := self.domain_prompts.get(self.cfg.domain_key):
                    return self._format_domain_prompt(domain_data)
            # 再尝试用 domain 名称字符串
            if domain_data := self.domain_prompts.get(self.cfg.medical_domain):
                return self._format_domain_prompt(domain_data)
            raise ValueError(
                f"The medical domain cannot be found in the configuration file: "
                f"{self.cfg.domain_key} / {self.cfg.medical_domain}"
            )

    # ====================
    # Phase 2-1: Domain-Specific Features (Steps 8-10,customized prompt)
    # ====================
    def create_custom_domain_agent(self, sct_content: str) -> str:
        """Create a new agent to assist in new domain-specific features"""
        messages = [
            {"role": "system", "content": """Please complete the following tasks:
                1. Classify these conversation transcriptions into six topics.
                2. Summarize the language expression features.
                3. Match each topic with the most typical cases from the provided conversation transcriptions.
                4. Return the results in plain text format."""},
            {"role": "user", "content": sct_content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.5
        )
        return "Customized domain-specific features"

    # ====================
    # Phase 2-2: Medical knowledge datasets (Steps 11-13)
    # ====================
    def load_knowledge(self) -> str:
        """Load all knowledge from the knowledge base"""
        knowledge_text = []
        
        required_files = [
            *glob.glob(f"{self.cfg.knowledge_base}/*.txt"),
            *glob.glob(f"{self.cfg.knowledge_base}/*.pdf")
        ]
        
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing knowledge base file: {missing}")

        for file_path in required_files:
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    knowledge_text.append(file.read())
            elif file_path.endswith(".pdf"):
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    knowledge_text.append(text)
        
        return (
            "These are the medical and dialectal knowledge relevant to medical conversation transcriptions "
            f"in the {self.cfg.medical_domain.upper()}. Please remember this information and use it to perform subsequent structured processing.\n"
            + "\n".join(knowledge_text)
        )

    # ====================
    # Phase 3-1: Noise removal prompts generation (Step 14, customize the word list)
    # ====================

    def _process_line_by_line(self, func, text: str) -> str:
        """逐行调用 func 处理，保持输入行数对齐"""
        lines = text.splitlines()
        processed = []
        for line in lines:
            if line.strip():
                try:
                    processed_line = func(line)
                except Exception:
                    processed_line = line  # 出错时保底
            else:
                processed_line = ""  # 保留空行
            processed.append(processed_line)
        return "\n".join(processed)

    def _process_in_chunks(self, func, text: str, chunk_size: int = 50) -> str:
        """
        按行切块，每块 chunk_size 行，用 func 处理。
        要求 func 接收一段多行文本并返回相同行数的输出。
        """
        lines = text.splitlines()
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        processed_chunks = []

        for chunk in chunks:
            chunk_text = "\n".join(chunk)
            processed_chunk = func(chunk_text)

            # 强制对齐：如果输出行数和输入不一致，补空行或截断
            out_lines = processed_chunk.splitlines()
            if len(out_lines) < len(chunk):
                out_lines += [""] * (len(chunk) - len(out_lines))
            elif len(out_lines) > len(chunk):
                out_lines = out_lines[:len(chunk)]

            processed_chunks.append("\n".join(out_lines))

        return "\n".join(processed_chunks)

    # def build_wordlist(self):
    #     """Customize the word list"""
    #     os.environ["GLOG_minloglevel"] = "2"
    #     # Read SCT.txt file
    #     with open(self.cfg.sct_file, 'r', encoding='utf-8') as f:
    #         text = f.read()
    #
    #     # Dynamically adjust vocab_size
    #     unique_tokens = len(set(text.split()))
    #     vocab_size = min(self.cfg.vocab_size, unique_tokens)
    #     print(f"Adjusted vocab_size: {vocab_size} (unique tokens: {unique_tokens})")
    #
    #     # Using temporary files
    #     with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
    #         temp_file.write(text)
    #         temp_file_path = temp_file.name
    #
    #     try:
    #         # Training the sentencepiece model
    #         spm.SentencePieceTrainer.train(
    #             input=temp_file_path,
    #             model_prefix='medical_spm',
    #             vocab_size=800,
    #             model_type='unigram',
    #             character_coverage=0.9995
    #         )
    #
    #         # Load the trained model
    #         sp = spm.SentencePieceProcessor()
    #         sp.load('medical_spm.model')
    #
    #         # Return wordlist
    #         return [sp.id_to_piece(id) for id in range(sp.get_piece_size())][:vocab_size]
    #
    #     finally:
    #         os.remove(temp_file_path)

    def _add_line_numbers(self, text: str) -> str:
        """为每行添加唯一行号"""
        lines = text.strip().splitlines()
        return "\n".join([f"[{i + 1}] {line}" for i, line in enumerate(lines)])

    def _remove_line_numbers(self, text: str) -> str:
        """去除输出中的行号"""
        lines = text.strip().splitlines()
        cleaned = [re.sub(r"^\[\d+\]\s*", "", line.strip()) for line in lines]
        return "\n".join(cleaned)

    def _restore_missing_lines(self, original_text: str, model_output: str) -> str:
        """检查并补齐缺失行"""
        orig_lines = original_text.strip().splitlines()
        out_lines = model_output.strip().splitlines()

        # 解析模型输出中的编号
        out_dict = {}
        for line in out_lines:
            match = re.match(r"^\[(\d+)\]", line)
            if match:
                out_dict[int(match.group(1))] = line
        # 补齐缺失
        restored = []
        for i in range(len(orig_lines)):
            if i + 1 in out_dict:
                restored.append(out_dict[i + 1])
            else:
                restored.append(f"[{i + 1}] [Line missing - restored as blank]")
        return "\n".join(restored)

    def build_wordlist(self):
        """兼容多平台"""
        os.environ["GLOG_minloglevel"] = "2"
        with open(self.cfg.sct_file, 'r', encoding='utf-8') as f:
            text = f.read()

        unique_tokens = len(set(text.split()))
        vocab_size = max(800, min(self.cfg.vocab_size, unique_tokens))
        print(f"Adjusted vocab_size: {vocab_size} (unique tokens: {unique_tokens})")

        import tempfile, time
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix=".txt") as temp_file:
            temp_file.write(text)
            temp_file_path = temp_file.name

        try:
            print("Starting SentencePiece training...")
            model_prefix = os.path.join(self.cfg.output_dir, "medical_spm")
            spm.SentencePieceTrainer.train(
                input=temp_file_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type='unigram',
                character_coverage=0.9995
            )
            print("Training completed, loading model...")
            sp = spm.SentencePieceProcessor()
            sp.load(model_prefix + ".model")
            print("Model loaded successfully.")
            wordlist = [sp.id_to_piece(i) for i in range(sp.get_piece_size())][:vocab_size]
            print(f"Wordlist built ({len(wordlist)} entries).")
            return wordlist
        finally:
            for _ in range(3):
                try:
                    os.remove(temp_file_path)
                    break
                except PermissionError:
                    time.sleep(0.5)

    # ====================
    # Phase 3-1: Noise removal prompts generation (Step 15, noise removal)
    # ====================

    def _remove_noise_chunk(self, text: str) -> str:
        """一次处理多行（chunk），保持行数一致"""
        numbered_text = self._add_line_numbers(text)
        messages = [
            {"role": "user", "content": f"""
    You are a medical conversation cleaner.  
    Your goal is to **aggressively remove irrelevant or noisy content**, keeping only text that clearly belongs to a real medical conversation.

    ### Data Format
    [ID] [speaker label]: [content]

    ### What Counts as Noise
    - **Hospital announcements** (e.g., “Please go to Room 1 for consultation”).
    - **One-sided phone calls** (e.g., “Hello… okay… bye.”).
    - **Non-medical speech**, meaningless chatter, or random interjections (“嗯”, “哎呀”, “对对对”, etc.).
    - **Gibberish**, incomplete, or fragmented text with no clear medical meaning.
    - **Background staff or unrelated talk**.
    - **Duplicate or repeated meaningless phrases**.

    ### How to Process
    - Be **bold and decisive** — if a line **seems mostly irrelevant**, treat it as noise.
    - If the medical intent or logic **cannot be clearly inferred**, treat it as noise.
    - Replace **the [content] of any line that is mostly or entirely noise** with '[Noise removed]'.
    - If a line contains both noise and meaningful content, **keep only the meaningful medical part**, deleting the rest.
    - If a line is clearly part of a real doctor–patient dialogue, keep it unchanged.

    ### Format Rules (MUST FOLLOW)
    - Keep **exactly the same number of lines** as input.
    - **Preserve the [ID] tag** at the beginning of each line exactly as input.
    - Do **NOT** merge, split, reorder, or delete lines.
    - Keep [speaker label] (e.g., “Doctor:”, “Patient:”).
    - Output only cleaned lines, no explanations.
    - Do not change the language of the conversation.

    Be confident and aggressive: it is better to **delete questionable or unclear lines** than to keep irrelevant noise.  
    Now clean the following text:
    {numbered_text}
    """}
        ]

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
        output = self._restore_missing_lines(numbered_text, output)
        return self._remove_line_numbers(output)

    def _remove_noise(self, text: str) -> str:
        """分块去噪，默认每50行一个chunk"""
        return self._process_in_chunks(self._remove_noise_chunk, text, chunk_size=50)

    # def _remove_noise_line(self, line: str) -> str:
    #     """单行去噪"""
    #     messages = [
    #         {"role": "user", "content": f"""Please process the conversation transcriptions according to the following requirements:
    #                     Objective:
    #                     Remove noise from the medical conversation, such as hospital announcements or a one-side phone medical conversations. Leaves original sentence structure unchanged. Please keep the number of input lines consistent with that of output lines.
    #                     Method:
    #                     1. Identify Noise: When a segment of the medical conversations is interrupted, determine whether it is hospital announcements noise or phone medical conversations based on its content.
    #                     2. Hospital announcements content: Typically in the form of imperative sentences, with the same phrase repeated twice or more. The main content usually includes statements such as 'Please go to Room1 for consultation'.
    #                     3. One-side phone medical conversations: Typically only one person asks or answers a question, without corresponding response. Such as 'Hello, hi... okay... I understand... I will call you back shortly.'
    #                     \n{line}"""}
    #     ]
    #     response = self.client.chat.completions.create(
    #         model=self.cfg.model_name,
    #         messages=messages,
    #         temperature=0.1
    #     )
    #     return response.choices[0].message.content.strip()
    #
    # def _remove_noise(self, text: str) -> str:
    #     """逐行去噪，保持行数对齐"""
    #     return self._process_line_by_line(self._remove_noise_line, text)

    # def _remove_noise(self, text: str) -> str:
    #     """Noise removal"""
    #     messages = [
    #         {"role": "user", "content": f"""Please process the conversation transcriptions according to the following requirements:
    #             Objective:
    #             Remove noise from the medical conversation, such as hospital announcements or a one-side phone medical conversations. Leaves original sentence structure unchanged. Please keep the number of input lines consistent with that of output lines.
    #             Method:
    #             1. Identify Noise: When a segment of the medical conversations is interrupted, determine whether it is hospital announcements noise or phone medical conversations based on its content.
    #             2. Hospital announcements content: Typically in the form of imperative sentences, with the same phrase repeated twice or more. The main content usually includes statements such as 'Please go to Room1 for consultation'.
    #             3. One-side phone medical conversations: Typically only one person asks or answers a question, without corresponding response. Such as 'Hello, hi... okay... I understand... I will call you back shortly.'
    #             \n{text}"""}
    #     ]
    #
    #     response = self.client.chat.completions.create(
    #         model=self.cfg.model_name,
    #         messages=messages,
    #         temperature=0.1
    #     )
    #     return response.choices[0].message.content
    
    # ====================
    # Phase 3-2: Content error correction prompts generation (Step 16)
    # ====================
    def _correct_content_chunk(self, text: str) -> str:
        """一次处理多行文本，保持行数一致"""
        numbered_text = self._add_line_numbers(text)
        domain_prompt = self.get_domain_prompt()
        knowledge = self.load_knowledge()
        # {"role": "system", "content": f"{knowledge}\n\n{domain_prompt}"},
        messages = [
        {"role": "user", "content": f"""
    You are a medical transcription corrector.  
    Your task is to **actively correct, refine, and smooth** the following conversation so it reads like a coherent, natural doctor–patient dialogue.

    ### Data Format
    [ID] [speaker label]: [content]

    ### What You Must Fix
    1. **Medical terminology** — correct misheard or misspelled medical terms confidently.
    2. **Homophones or dialect words** — infer the intended words from context (e.g., “车线” → “拆线”).
    3. **Broken grammar or word order** — reorganize them into fluent spoken sentences.
    4. **Incomplete or unclear lines** — complete or correct them based on common medical reasoning.
    5. **Remove filler words** (e.g., “啊”, “嗯”, “就是”, “那个”) when excessive, while keeping a natural spoken tone.


    ### Format Rules (STRICT)
    - Keep **exactly the same number of lines** as input.
    - **Preserve each [ID] tag** exactly as input.
    - **Do NOT** merge, split, reorder, or delete lines.
    - Keep [speaker label] (e.g., “Doctor:”, “Patient:”).
    - Output only corrected conversation, no explanations.
    - Do not change the language of the conversation.

    Now correct and polish the following transcription:
    {numbered_text}
    """}
        ]

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
        output = self._restore_missing_lines(numbered_text, output)
        return self._remove_line_numbers(output)

    def _correct_content(self, text: str) -> str:
        """分块内容矫正，默认每50行一个chunk"""
        return self._process_in_chunks(self._correct_content_chunk, text, chunk_size=50)

    # def _correct_content_line(self, line: str) -> str:
    #     """单行内容矫正"""
    #     domain_prompt = self.get_domain_prompt()
    #     knowledge = self.load_knowledge()
    #     messages = [
    #         {"role": "system", "content": f"{knowledge}\n\n{domain_prompt}"},
    #         {"role": "user", "content": f"""Integrate the fragmented transcriptions into complete medical conversation transcriptions based on domain-specific features and medical knowledge. Make the following corrections accordingly:
    #                 Note:
    #                 1. Medical Terminology Correction: Due to potential issues with speech-to-text tools, medical terminology may be incorrectly transcribed. Verify and correct the medical terminology, but keep the colloquial expression without over-specialization.
    #                 2. Near-homophone Correction: If some words seem out of place in the medical conversations, they may be homophones or dialect-related errors. Based on context, domain-specific features and medical knowledge, infer and correct ambiguous terms. For case, '车（che）线'could be a misinterpretation of '拆（chai）线' when referring to surgical suture removal.
    #                 3. Correction Scope: Only correct individual words; do not change the original colloquial sentence structure or expression.
    #                 Ensure that the final output is a clear, logically consistent medical conversation transcription that preserves the original language, retains the features of natural spoken expression, leaves every original sentence structure unchanged, and please keep the number of input lines consistent with that of output lines.:\n{line}"""}
    #     ]
    #     response = self.client.chat.completions.create(
    #         model=self.cfg.model_name,
    #         messages=messages,
    #         temperature=0.1
    #     )
    #     return response.choices[0].message.content.strip()
    #
    # def _correct_content(self, text: str) -> str:
    #     """逐行内容矫正，保持行数对齐"""
    #     return self._process_line_by_line(self._correct_content_line, text)

    # def _correct_content(self, text: str) -> str:
    #     """Content error correction"""
    #     domain_prompt = self.get_domain_prompt()
    #     knowledge = self.load_knowledge()
    #
    #     messages = [
    #     {"role": "system", "content": f"{knowledge}\n\n{domain_prompt}"},
    #     {"role": "user", "content": f"""Integrate the fragmented transcriptions into complete medical conversation transcriptions based on domain-specific features and medical knowledge. Make the following corrections accordingly:
    #         Note:
    #         1. Medical Terminology Correction: Due to potential issues with speech-to-text tools, medical terminology may be incorrectly transcribed. Verify and correct the medical terminology, but keep the colloquial expression without over-specialization.
    #         2. Near-homophone Correction: If some words seem out of place in the medical conversations, they may be homophones or dialect-related errors. Based on context, domain-specific features and medical knowledge, infer and correct ambiguous terms. For case, '车（che）线'could be a misinterpretation of '拆（chai）线' when referring to surgical suture removal.
    #         3. Correction Scope: Only correct individual words; do not change the original colloquial sentence structure or expression.
    #         Ensure that the final output is a clear, logically consistent medical conversation transcription that preserves the original language, retains the features of natural spoken expression, leaves every original sentence structure unchanged, and please keep the number of input lines consistent with that of output lines.:\n{text}"""}
    #     ]
    #
    #     response = self.client.chat.completions.create(
    #         model=self.cfg.model_name,
    #         messages=messages,
    #         temperature=0.1
    #     )
    #     return response.choices[0].message.content
    # Identify the speaker for each line. Label as:
    
    # ====================
    # Phase 3-3: Speaker identification prompts generation (Step 17)
    # ====================

    def _identify_speakers_chunk(self, text: str) -> str:
        """一次处理多行文本，保持行数一致"""
        numbered_text = self._add_line_numbers(text)
        messages = [
            {"role": "system", "content": """
    You are a speaker labeler for medical dialogues.  
    Your task is to ignore the original [speaker label] and re-label the correct [speaker label] for each line based on the line's [content] and its context.
    Be confident.
    
    ### Data Format
    [ID] [speaker label]: [content]

    ### Valid Labels
    Doctor:
    Patient:
    Others:

    ### How to Decide
    - “Doctor” gives medical advice, asks examination questions, or provides explanations.
    - “Patient” describes symptoms, asks for guidance, or expresses concerns.
    - “Others” include family members, assistants, or background staff.
    - If the [content] of this line is '[Noise removed]', label it as "Others".

    ### Behavior Rules
    - Do not hesitate to label roles when content clearly implies another speaker.

    ### Format Rules (STRICT)
    - Keep **exactly the same number of lines** as input.
    - Preserve every [ID] exactly as input.
    - Do **NOT** merge, split, delete, or reorder lines.
    - Keep the **original content after the label** unchanged.
    - Output only relabeled lines, nothing else.
    """}
            ,
            {"role": "user", "content": numbered_text}
        ]

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
        output = self._restore_missing_lines(numbered_text, output)
        return self._remove_line_numbers(output)

    def _identify_speakers(self, text: str) -> str:
        """分块说话人识别，默认每50行一个chunk"""
        return self._process_in_chunks(self._identify_speakers_chunk, text, chunk_size=50)

    # def _identify_speakers(self, text: str) -> str:
    #     """Speaker identification"""
    #     messages = [
    #         {"role": "system", "content": "It is possible that part of the speaker identification in this content is incorrect. Identify the speaker on a line-by-line basis, based on the content of the medical conversation transcriptions, and label it as: Doctor: / Patient: / Others: , without altering the original sentence structure, and please keep the number of input lines consistent with that of output lines."},
    #         {"role": "user", "content": text}
    #     ]
    #
    #     response = self.client.chat.completions.create(
    #         model=self.cfg.model_name,
    #         messages=messages,
    #         temperature=0.1
    #     )
    #     return response.choices[0].message.content

    # ====================
    # Phase 3-4: Segmentation
    # ====================
    def apply_segmentation(self, text: str, strategy: str = "Topic-based", chunk_size: int = 50) -> str:
        """Apply the specified segmentation strategy with optional chunk-based processing."""
        if not hasattr(self, "segmentation_strategies"):
            self.segmentation_strategies = {
                "Topic-based": self._segment_by_topic,
                "Chronological": self._segment_by_chronological,
                "Emotion-based": self._segment_by_emotion
            }

        if strategy not in self.segmentation_strategies:
            raise ValueError(f"Invalid segmentation strategy: {strategy}")

        segment_func = self.segmentation_strategies[strategy]
        return self._process_in_chunks(segment_func, text, chunk_size=chunk_size)

    def remove_timestamps(self, text: str) -> str:
        """Remove timestamping patterns such as '00:12' or '12:34:56'."""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r'^\d+:\d+(?::\d+)?\s*', '', line.strip())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines)

    # ====================
    # Topic-based segmentation
    # ====================
    def _segment_by_topic(self, text: str) -> str:
        """Topic-based segmentation"""
        cleaned_text = self.remove_timestamps(text)
        system_prompt = f"""
        Domain: {self.cfg.medical_domain}
        Topic list: {self._get_topic_templates()}
        
        Identify turning points in topics based on context and segment the medical conversation accordingly.
        Each topic should have a concise summarizing phrase. Return the results in the format:
        Topic 1: [Topic name]
        [Conversation]
        Topic 2: [Topic name]
        [Conversation]
        """
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cleaned_text}
            ],
            temperature=0.3
        )
        return self._validate_topic_format(response.choices[0].message.content)

    def _get_topic_templates(self) -> str:
        """Return topic templates based on the medical domain from Domain Specific Prompts.json"""
        domain_name = self.cfg.medical_domain.strip()
        if not hasattr(self, "domain_prompts") or not self.domain_prompts:
            raise RuntimeError("Domain prompts have not been loaded yet.")

        # 优先匹配 Domain Specific Prompts.json 中的结构
        domain_data = self.domain_prompts.get(domain_name)
        if domain_data and "topics" in domain_data:
            topic_list = list(domain_data["topics"].keys())
            return ", ".join([t.replace(":", "").strip() for t in topic_list])

        # 如果没找到匹配（例如自定义 domain）
        # 做一次模糊匹配，兼容 like “Surgery” vs “Surgical procedures”
        lower_name = domain_name.lower()
        if "surg" in lower_name:
            fallback = ["Preoperative preparation", "Surgical procedure", "Postoperative care"]
        elif "hospital" in lower_name:
            fallback = ["Admission guidance", "Discharge management", "Medication precautions"]
        elif "checkup" in lower_name:
            fallback = ["Examination preparation", "Health counseling", "Result interpretation"]
        else:
            fallback = ["Symptom inquiry", "Medication guidance", "Follow-up arrangement"]

        return ", ".join(fallback)

    def _validate_topic_format(self, text: str) -> str:
        """Ensure topic-based segmentation is properly formatted."""
        pattern = r"(Topic \d+: .+?)\n((?:.+\n)+?)(?=Topic \d+:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        return "\n\n".join([f"{title}\n{content.strip()}" for title, content in matches])

    # ====================
    # Chronological segmentation
    # ====================
    def _segment_by_chronological(self, text: str) -> str:
        """Chronological segmentation"""
        dialogues = self._parse_chronological_dialogues(text)
        segments, current_segment, prev_time = [], [], None
        for turn in dialogues:
            if prev_time and (turn["timestamping"] - prev_time) > 1:
                if len(current_segment) >= 3:
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append(turn)
            prev_time = turn["timestamping"]
        if current_segment:
            segments.append(current_segment)
        return self._format_chronological_segments(segments)

    def _parse_chronological_dialogues(self, text: str) -> List[Dict]:
        """Parse conversation lines with timestamps"""
        dialogues = []
        for line in text.split('\n'):
            if match := re.match(r'^(\S+)\s+(\d+:\d+:\d+)\s+(.*)', line):
                role, ts, content = match.groups()
                dialogues.append({
                    "role": role,
                    "timestamping": self._chronological_to_seconds(ts),
                    "content": content
                })
        return dialogues

    def _chronological_to_seconds(self, time_str: str) -> float:
        """Convert HH:MM:SS or MM:SS to seconds."""
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return 0

    def _format_chronological_segments(self, segments: List[List[Dict]]) -> str:
        """Format chronological segmentation results"""
        output = []
        for idx, seg in enumerate(segments, 1):
            start = datetime.fromtimestamp(seg[0]["timestamping"]).strftime("%H:%M:%S")
            end = datetime.fromtimestamp(seg[-1]["timestamping"]).strftime("%H:%M:%S")
            output.append(f"Segment {idx} ({start} - {end}):")
            output.extend([f"{turn['role']}: {turn['content']}" for turn in seg])
            output.append("")
        return "\n".join(output)

    # ====================
    # Emotion-based segmentation
    # ====================
    def _load_emotion_data(self, file_path: str) -> dict:
        """Load emotion features JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[Warning] Emotion data file '{file_path}' not found.")
            return {}

    def _enhance_emotion_tags(self, text: str) -> str:
        """Enhance emotional expressions in patient speech."""
        if self.emotion_data is None:
            self.emotion_data = self._load_emotion_data("Features of patient emotions.json")
        if not self.emotion_data:
            raise ValueError("Emotion data not loaded. Please check 'Features of patient emotions.json'.")
        emotion_prompt = "\n".join([
            f"{emo}: {info['Features']} Example: {', '.join(info['Examples'])}"
            for emo, info in self.emotion_data.items()
        ])
        enhancement_prompt = (
            "Based on the six basic emotions, enhance the patient's emotional expression. "
            "Modify punctuation, add interjections, or include brief annotations. "
            "Do not alter semantic content.\n\n"
            f"{emotion_prompt}"
        )
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[
                {"role": "system", "content": enhancement_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content

    def _segment_by_emotion(self, text: str) -> str:
        """Emotion-based segmentation."""
        cleaned_text = self.remove_timestamps(text)
        enhanced_text = self._enhance_emotion_tags(cleaned_text)
        prompt = (
            "Segment the conversation based on changes in the patient's emotional state. "
            "Ensure completeness of each segment.\n\n"
            f"{enhanced_text}"
        )
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": cleaned_text}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content

    # ====================
    # Phase 4: Generation of SCT (Steps 24-28)
    # ====================
    def process_segment(self, segment: str) -> str:
        """Processing RUCT batches"""
        # Noise removal
        print("Start noise removal...")
        cleaned = self._remove_noise(segment)
        print("Noise removal is finished!")
        
        # Content correction
        print("Start content correction...")
        corrected = self._correct_content(cleaned)
        print("Content correction is finished!")
        
        # Speaker identification
        print("Start speaker identification...")
        speaker_identified = self._identify_speakers(corrected)
        print("Speaker identification is finished!")

        return self.apply_segmentation(speaker_identified)

    # ====================
    # Output
    # ====================
    def generate_output(self, processed_segments: List[str]) -> str:
        """Output"""
        return "\n\n".join(processed_segments)

    def run_pipeline(self):
        """Run procedures based on the selected medical domain"""
        # Create output directory
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            # 1. Pre-segment RUCT
            print("Start pre-segmenting RUCT...")
            segments = self.preprocess_ruct()
            print("Pre-segmenting is finished!")

            # 2. Generate custom prompt if domain is 5
            if self.cfg.medical_domain == "Customized domain-specific features":
                print("Generating custom prompt...")
                custom_prompt = self.create_custom_domain_agent(self.cfg.sct_file)
                print("Custom prompt generated!")
                self.domain_prompts["Customized domain-specific features"] = custom_prompt

            # 3. Processing steps with user interaction
            for idx, seg in enumerate(segments, 1):
                print(f"\nProcessing batch {idx}/{len(segments)}:")

                # Noise Removal
                print("- Noise removal...")
                cleaned = self._remove_noise(seg)
                if not self._user_continue(): return self._save_and_exit()

                # Content Correction
                print("- Content correction...")
                corrected = self._correct_content(cleaned)
                if not self._user_continue(): return self._save_and_exit()

                # Speaker Identification
                print("- Speaker identification...")
                speaker_identified = self._identify_speakers(corrected)
                if not self._user_continue(): return self._save_and_exit()

                self._interim_results.append(speaker_identified)

            # 4. Ask user if they want to proceed to segmentation strategy selection
            print("\nAll batches processed. ")
            if not self._user_continue():  
                return self._save_and_exit()

            # 5. Select segmentation strategy
            selected_strategy = "Topic-based"
            print(f"\nApplying {selected_strategy} segmentation strategy...")

            # 6. Generation of SCT
            processed = []
            for idx, interim in enumerate(self._interim_results, 1):
                print(f"\nProcessing batch {idx}/{len(self._interim_results)}:")
                processed.append(self.apply_segmentation(interim, selected_strategy))

            # 7. Save final output
            final_output = self.generate_output(processed)
            output_path = Path(self.cfg.output_dir) / "processed_sct.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_output)
            print(f"\nProcessing finished! SCT saved to: {output_path}")

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise

    def _user_continue(self):
        """Automatically continue without user interaction"""
        return True

    def _save_and_exit(self):
        """Save current progress and exit"""
        output_path = Path(self.cfg.output_dir) / "interim_sct.txt"
        interim_output = self.generate_output(self._interim_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(interim_output)
        print(f"Partial processing saved to: {output_path}. Exiting now.")
        os._exit(0) 


# ====================
# Main
# ====================
if __name__ == "__main__":
    config = Config()
    pipeline = MedicalTranscriptionPipeline(config)

    pipeline.run_pipeline()
