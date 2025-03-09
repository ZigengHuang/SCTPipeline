import os
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
import openai

# ====================
# Set the parameters
# ====================
class Config:
    def __init__(self):
        # API
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")
        self.model_name = "deepseek-reasoner"
        self.temperature = 0.1
        
        # file path
        self.input_ruct = "./data/input_ruct.txt"
        self.output_dir = "./processed"
        self.knowledge_base = "./medical_knowledge"
        
        # processing parameters
        self.medical_domain = "General Outpatient Visits"
        self.vocab_size = 5000
        self.similarity_threshold = 0.9

        # segementation strategies
        self.segmentation_strategy = 'topic'  # (Optional)'topic'/'Chronological'/'emotion'


# ====================
# Procedures
# ====================
class MedicalTranscriptionPipeline:
    def __init__(self, config: Config):
        self.cfg = config
        self.client = OpenAI(
            api_key=self.cfg.api_key,
            base_url="https://api.deepseek.com"
        )
        
        # initialization
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        self.segmentation_strategies = {
            'topic': self._segment_by_topic,
            'Chronological': self._segment_by_Chronological,
            'emotion': self._segment_by_emotion
        }

        
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
    # Phase 2-1: Domain-Specific Features (Steps 7-10)
    # ====================
    def get_domain_prompt(self) -> str:
        """Select predefined domain-specific prompts"""
        domain_prompts = {
            "Health Checkups": "Prompt 1",
            "General Outpatient Visits": "Prompt 2",
            "Surgical Procedures":"Prompt 3",
            "Hospitalization Management": "Prompt4"
        }
        if self.cfg.medical_domain not in domain_prompts:
            raise ValueError(f"Invalid medical domain: {self.cfg.medical_domain}")
        return domain_prompts[self.cfg.medical_domain]

    # ====================
    # Phase 2-2: Medical knowledge datasets (Steps 11-13)
    # ====================
    def load_knowledge(self) -> str:
        """Medical knowledge datasets"""
        required_files = [
            *glob.glob(f"{self.cfg.knowledge_base}/general/*.txt"),
            *glob.glob(f"{self.cfg.knowledge_base}/department/*.txt")
        ]
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"Lack of knowledge datasets: {missing}")
        
        return (
            "These are the medical and dialectal knowledge relevant to medical conversation transcriptions "
            f"in the {self.cfg.medical_domain.upper()}. Please remember this information and use it to perform subsequent structured processing"
        )

    # ====================
    # Phase 3-1: Noise removal prompts generation (Steps 14-15)
    # ====================
    def build_vocab(self, text: str) -> List[str]:
        """Customize the word list"""
        temp_file = Path(self.cfg.output_dir) / "temp_corpus.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        spm.SentencePieceTrainer.train(
            input=str(temp_file),
            model_prefix='medical_spm',
            vocab_size=self.cfg.vocab_size,
            model_type='unigram',
            character_coverage=0.9995
        )
        
        sp = spm.SentencePieceProcessor()
        sp.load('medical_spm.model')
        return [sp.id_to_piece(id) for id in range(sp.get_piece_size())][:self.cfg.vocab_size]
    
    
    def _remove_noise(self, text: str) -> str:
        """Noise removal"""
        messages = [
            {"role": "system", "content": self.get_domain_prompt()},
            {"role": "user", "content": f"Please process the conversation transcriptions according to the following requirements:
                Objective:
                Remove noise from the medical conversation, such as hospital announcements or a one-side phone medical conversations.
                Method:
                1、Identify Noise: When a segment of the medical conversations is interrupted, determine whether it is hospital announcements noise or phone medical conversations based on its content.
                2、Hospital announcements content: Typically in the form of imperative sentences, with the same phrase repeated twice or more. The main content usually includes statements such as 'Please go to Room1 for consultation'.
                3、One-side phone medical conversations: Typically only one person asks or answers a question, without corresponding response. Such as 'Hello, hi... okay... I understand... I will call you back shortly.'：\n{text}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    # ====================
    # Phase 3-2: Content error correction prompts generation (Step 16)
    # ====================
    def _correct_content(self, text: str) -> str:
        """Content error correction"""
        knowledge = self.load_knowledge()
        messages = [
            {"role": "system", "content": knowledge},
            {"role": "user", "content": f"Integrate the fragmented transcriptions into complete medical conversation transcriptions based on domain-specific features and medical knowledge. Make the following corrections accordingly: 
                Note:
                1. Medical Terminology Correction: Due to potential issues with speech-to-text tools, medical terminology may be incorrectly transcribed. Verify and correct the medical terminology, but keep the colloquial expression without over-specialization.
                2. Near-homophone Correction: If some words seem out of place in the medical conversations, they may be homophones or dialect-related errors. Based on context, domain-specific features and medical knowledge, infer and correct ambiguous terms. For case, '车（che）线'could be a misinterpretation of '拆（chai）线' when referring to surgical suture removal.
                3. Correction Scope: Only correct individual words; do not alter sentence structure or expressions.
                Ensure that the final output is a clear, logically consistent medical conversation transcriptions that retains the features of natural spoken expression.:\n{text}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    # ====================
    # Phase 3-3: Speaker identification prompts generation (Step 17)
    # ====================
    def _identify_speakers(self, text: str) -> str:
        """Speaker identification"""
        messages = [
            {"role": "system", "content": "Identity the speaker on a line-by-line basis, based on the content of the medical conversation transcriptions, and label it as: Doctor: / Patient: / Others:"},
            {"role": "user", "content": text}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    # ====================
    # Phase 3-4: Segmentation prompts generation (Step 18-24)
    # ====================
    def apply_segmentation(self, text: str, strategy: str) -> str:
        """ Select one segementation strategy based on requirements"""
        if strategy not in self.segmentation_strategies:
            raise ValueError(f"Invalid segmentation strategy, optional: {list(self.segmentation_strategies.keys())}")
        return self.segmentation_strategies[strategy](text)

    def _segment_by_topic(self, text: str) -> str:
        """Topic-based"""
        system_prompt = f"""
Domain: {self.cfg.medical_domain}
topic list: {self._get_topic_templates()}

Identify the turning points in the topics based on context and segment the medical conversation transcriptions accordingly. Ensure the segmented medical conversation transcriptions are complete. Each identified topic should be represented as a short summarizing phrase. Return the medical conversation transcriptions in the following plain text format:
Topic 1: [Topic name]
[Complete medical conversation transcriptions]
Topic 2: [Topic name]
[Complete medical conversation transcriptions]"""
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return self._validate_topic_format(response.choices[0].message.content)

    def _get_topic_templates(self) -> str:
        """Detailed topic list"""
        topic_templates = {
            "General": ["Location inquiries", "Missing item reports", "Insurance inquiries"], # Only show part of the list
            "Surgery": ["Pre-procedure testing", "Anesthesia consultation and instructions", "Explaining procedure risks and benefits"] # Only show part of the list
        }
        return ", ".join(topic_templates.get(self.cfg.medical_domain, topic_templates["General"]))

    def _validate_topic_format(self, text: str) -> str:
        """Formatting topic-based segmentation results"""
        pattern = r"(Topic \d+: .+?)\n((?:.+\n)+?)(?=Topic \d+:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        return "\n\n".join([f"{title}\n{content.strip()}" for title, content in matches])

    def _segment_by_Chronological(self, text: str) -> str:
        """Chronological"""
        dialogues = self._parse_Chronological_dialogues(text)
        segments = []
        current_segment = []
        prev_Chronological = None
        
        for turn in dialogues:
            if prev_Chronological and (turn["timestamping"] - prev_Chronological) > 1:
                if len(current_segment) >= 3:  # at least 3 round of dialog
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append(turn)
            prev_Chronological = turn["timestamping"]
        
        if current_segment:
            segments.append(current_segment)
        return self._format_Chronological_segments(segments)

    def _parse_Chronological_dialogues(self, text: str) -> List[Dict]:
        """Parsing timestamping conversation transcriptions"""
        dialogues = []
        for line in text.split('\n'):
            if match := re.match(r'^(\S+)\s+(\d+:\d+:\d+)\s+(.*)', line):
                role, ts, content = match.groups()
                dialogues.append({
                    "role": role,
                    "timestamping": self._Chronological_to_seconds(ts),
                    "content": content
                })
        return dialogues

    def _Chronological_to_seconds(self, Chronological_str: str) -> float:
        """Format conversion"""
        parts = list(map(int, Chronological_str.split(':')))
        if len(parts) == 2:  # MM:SS
            return parts[0]*60 + parts[1]
        elif len(parts) == 3:  # HH:MM:SS
            return parts[0]*3600 + parts[1]*60 + parts[2]
        return 0

    def _format_Chronological_segments(self, segments: List[List[Dict]]) -> str:
        """Formatting chronological segmentation results"""
        output = []
        for idx, seg in enumerate(segments, 1):
            start = datetime.fromtimestamp(seg[0]["timestamping"]).strftime("%H:%M:%S")
            end = datetime.fromtimestamp(seg[-1]["timestamping"]).strftime("%H:%M:%S")
            output.append(f"Segment {idx} ({start} - {end}):")
            output.extend([f"{turn['role']}: {turn['content']}" for turn in seg])
            output.append("")
        return '\n'.join(output)
    
    def _process_emotions(self, text: str) -> str:
        """Emotion-based"""
        messages = [
            {"role": "system", "content": "Please label according to the six basic emotions"},
            {"role": "user", "content": text}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.5
        )
        return response.choices[0].message.content

    def _segment_by_emotion(self, text: str) -> str:
        """Emotional expression features"""
        dialogues = self._parse_emotional_dialogues(text)
        enhanced = self._enhance_emotion_tags(dialogues)
        segments = []
        current_segment = []
        prev_emotion = None
        
        for turn in enhanced:
            if turn.get("emotion") != prev_emotion and current_segment:
                segments.append(current_segment)
                current_segment = []
            current_segment.append(turn)
            prev_emotion = turn.get("emotion")
        
        if current_segment:
            segments.append(current_segment)
        return self._format_emotion_segments(segments)

    def _parse_emotional_dialogues(self, text: str) -> List[Dict]:
        """Identifying emotions"""
        dialogues = []
        for line in text.split('\n'):
            if match := re.match(r'^\[(.*?)\]\s*(.*?)(?:\s*\((.*?)\))?$', line):
                role, content, emotion = match.groups()
                dialogues.append({
                    "role": role,
                    "content": content,
                    "emotion": emotion or "Neutral"
                })
        return dialogues

    def _enhance_emotion_tags(self, dialogues: List[Dict]) -> List[Dict]:
        """Emotion annotations"""
        for turn in dialogues:
            if turn["role"] == "Patient":
                turn["content"] = f"{turn['content']} ({turn['emotion']})"
        return dialogues

    def _format_emotion_segments(self, segments: List[List[Dict]]) -> str:
        """Formatting emotion-based segmentation results"""
        output = []
        for idx, seg in enumerate(segments, 1):
            emotion = seg[0].get("emotion", "neutral")
            output.append(f"Emotion Segment {idx} ({emotion}):")
            output.extend([f"{turn['role']}: {turn['content']}" for turn in seg])
            output.append("")
        return '\n'.join(output)

    # ====================
    # Phase 4: Generation of SCT (Steps 25-30)
    # ====================
    def process_segment(self, segment: str, strategy: str = 'topic') -> str:
        """Processing RUCT batches"""
        cleaned = self._remove_noise(segment)
        corrected = self._correct_content(cleaned)
        speaker_identified = self._identify_speakers(corrected)
        emotion_enhanced = self._process_emotions(speaker_identified)
        return self.apply_segmentation(emotion_enhanced, strategy)


    
    # ====================
    # Output
    # ====================
    def generate_output(self, processed_segments: List[str]) -> str:
        """Output"""
        return "\n\n".join(processed_segments)

    def run_pipeline(self, segmentation_strategy: str = 'topic'):
        """Run procedures"""
        # create output dir
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. preprocessing
            print("Start pre-segmenting RUCT...")
            segments = self.preprocess_ruct()
            
            # 2. process each chunk
            processed = []
            for idx, seg in enumerate(segments, 1):
                print(f"Processing No. {idx}/{len(segments)} batch...")
                processed.append(self.process_segment(seg, segmentation_strategy))
            
            # 3. final output
            final_output = self.generate_output(processed)
            output_path = Path(self.cfg.output_dir) / "processed_sct.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_output)

            print(f"Finish processing！Save SCT to：{output_path}")
            
        except Exception as e:
            print(f"Errors in processing：{str(e)}")
            raise

# ====================
# Main
# ====================
if __name__ == "__main__":
    # initialize
    config = Config()
    config.segmentation_strategy = 'topic'  # Select one segmentation strategy: topic/Chronological/emotion
    
    # run pipeline
    pipeline = MedicalTranscriptionPipeline(config)
    pipeline.run_pipeline()