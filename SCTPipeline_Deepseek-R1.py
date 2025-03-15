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
import keyboard
import tempfile

# ====================
# Set the parameters
# ====================
class Config:
    def __init__(self):
        # API
        self.api_key = "Your API Key" 
        self.model_name = "deepseek-reasoner"
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
        args = parser.parse_args()

        # File path
        self.input_ruct = os.path.normpath(args.input)
        self.output_dir = os.path.normpath(args.output)

        # Mapping of departments and medical knowledge base paths
        self.domain_mapping = {
            1: ("Health checkups", "Health checkups", "./SCTPipeline/medical_knowledge_backup/Health checkups medical_knowledge"),
            2: ("Routine outpatient guidance", "Routine outpatient guidance", "./SCTPipeline/medical_knowledge_backup/Routine outpatient guidance medical_knowledge"),
            3: ("Surgery", "Surgery", "./SCTPipeline/medical_knowledge_backup/Surgery medical_knowledge"),
            4: ("Hospitalization guidance", "Hospitalization guidance", "./SCTPipeline/medical_knowledge_backup/Hospitalization guidance medical_knowledge"),
            5: ("Customized domain-specific features", "Customized domain-specific features", "./SCTPipeline/medical_knowledge_backup/General medical knowledge")
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
            api_key="Your API Key",
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
                return json.load(f)
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
            if domain_data := self.domain_prompts.get(self.cfg.domain_key):
                return self._format_domain_prompt(domain_data)
            raise ValueError(f"The medical domain cannot be found in the configuration file：{self.cfg.domain_key}")
        
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
    def build_wordlist(self):
        """Customize the word list"""
        os.environ["GLOG_minloglevel"] = "2"
        # Read SCT.txt file
        with open(self.cfg.sct_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Dynamically adjust vocab_size
        unique_tokens = len(set(text.split()))  
        vocab_size = min(self.cfg.vocab_size, unique_tokens)  
        print(f"Adjusted vocab_size: {vocab_size} (unique tokens: {unique_tokens})")
        
        # Using temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(text)
            temp_file_path = temp_file.name 
        
        try:
            # Training the sentencepiece model
            spm.SentencePieceTrainer.train(
                input=temp_file_path, 
                model_prefix='medical_spm',
                vocab_size=800, 
                model_type='unigram',
                character_coverage=0.9995
            )
            
            # Load the trained model
            sp = spm.SentencePieceProcessor()
            sp.load('medical_spm.model')
            
            # Return wordlist
            return [sp.id_to_piece(id) for id in range(sp.get_piece_size())][:vocab_size]
        
        finally:
            os.remove(temp_file_path)
            
    # ====================
    # Phase 3-1: Noise removal prompts generation (Step 15, noise removal)
    # ====================
    def _remove_noise(self, text: str) -> str:
        """Noise removal"""
        messages = [
            {"role": "user", "content": f"""Please process the conversation transcriptions according to the following requirements:
                Objective:
                Remove noise from the medical conversation, such as hospital announcements or a one-side phone medical conversations.
                Method:
                1、Identify Noise: When a segment of the medical conversations is interrupted, determine whether it is hospital announcements noise or phone medical conversations based on its content.
                2、Hospital announcements content: Typically in the form of imperative sentences, with the same phrase repeated twice or more. The main content usually includes statements such as 'Please go to Room1 for consultation'.
                3、One-side phone medical conversations: Typically only one person asks or answers a question, without corresponding response. Such as 'Hello, hi... okay... I understand... I will call you back shortly.'
                \n{text}"""}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    # ====================
    # Phase 3-2: Content error correction prompts generation (Step 16)
    # ====================
    def _correct_content(self, text: str) -> str:
        """Content error correction"""
        domain_prompt = self.get_domain_prompt()
        knowledge = self.load_knowledge()

        messages = [
        {"role": "system", "content": f"{knowledge}\n\n{domain_prompt}"},
        {"role": "user", "content": f"""Integrate the fragmented transcriptions into complete medical conversation transcriptions based on domain-specific features and medical knowledge. Make the following corrections accordingly: 
            Note:
            1. Medical Terminology Correction: Due to potential issues with speech-to-text tools, medical terminology may be incorrectly transcribed. Verify and correct the medical terminology, but keep the colloquial expression without over-specialization.
            2. Near-homophone Correction: If some words seem out of place in the medical conversations, they may be homophones or dialect-related errors. Based on context, domain-specific features and medical knowledge, infer and correct ambiguous terms. For case, '车（che）线'could be a misinterpretation of '拆（chai）线' when referring to surgical suture removal.
            3. Correction Scope: Only correct individual words; do not change the original colloquial sentence structure or expression.
            Ensure that the final output is a clear, logically consistent medical conversation transcriptions that retains the features of natural spoken expression.:\n{text}"""}
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
            {"role": "system", "content": "The speaker identification of this content is wrong. Identity the speaker on a line-by-line basis, based on the content of the medical conversation transcriptions, and label it as: Doctor: / Patient: / Others:"},
            {"role": "user", "content": text}
        ]
        
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    # ====================
    # Phase 3-4: Segmentation prompts generation (Steps 18-24)
    # ====================
    def _interactive_select_segmentation(self) -> str:
            """Choose the segmentation strategy"""
            options = ['Topic-based', 'Chronological', 'Emotion-based']
            print("\n" + "="*30)
            print("Please choose the segmentation strategy：")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            while True:
                choice = input("Please enter the option number (1-3，Default 1): ").strip()
                try:
                    if 1 <= int(choice) <= len(options):
                        selected = options[int(choice)-1]
                        print(f"Chosen：{selected}")
                        return selected
                    else:
                        print("Number out of range. Please re-enter.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

    def apply_segmentation(self, text: str, strategy: str) -> str:
        """Apply the segmentation strategy"""
        if strategy not in self.segmentation_strategies:
            raise ValueError(f"Invalid segmentation strategy: {strategy}")
        return self.segmentation_strategies[strategy](text)

    def remove_timestamps(self, text: str) -> str:
        """Remove timestamping"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = re.sub(r'^\d+:\d+\s*', '', line.strip())
            if cleaned_line: 
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    # ====================
    # Topic-based segmentation (Step 18)
    # ====================
    def _segment_by_topic(self, text: str) -> str:
        """Topic-based"""
        cleaned_text = self.remove_timestamps(text)

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
                {"role": "user", "content": cleaned_text}
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

    # ====================
    # Chronological segmentation (Steps 19-20)
    # ====================
    def _segment_by_chronological(self, text: str) -> str:
        """Chronological"""
        dialogues = self._parse_chronological_dialogues(text)
        segments = []
        current_segment = []
        prev_chronological = None
        
        for turn in dialogues:
            if prev_chronological and (turn["timestamping"] - prev_chronological) > 1:
                if len(current_segment) >= 3: 
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append(turn)
            prev_chronological = turn["timestamping"]
        
        if current_segment:
            segments.append(current_segment)
        return self._format_chronological_segments(segments)

    def _parse_chronological_dialogues(self, text: str) -> List[Dict]:
        """Parsing timestamping conversation transcriptions"""
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

    def _chronological_to_seconds(self, chronological_str: str) -> float:
        """Format conversion"""
        parts = list(map(int, chronological_str.split(':')))
        if len(parts) == 2:  # MM:SS
            return parts[0]*60 + parts[1]
        elif len(parts) == 3:  # HH:MM:SS
            return parts[0]*3600 + parts[1]*60 + parts[2]
        return 0

    def _format_chronological_segments(self, segments: List[List[Dict]]) -> str:
        """Formatting chronological segmentation results"""
        output = []
        for idx, seg in enumerate(segments, 1):
            start = datetime.fromtimestamp(seg[0]["timestamping"]).strftime("%H:%M:%S")
            end = datetime.fromtimestamp(seg[-1]["timestamping"]).strftime("%H:%M:%S")
            output.append(f"Segment {idx} ({start} - {end}):")
            output.extend([f"{turn['role']}: {turn['content']}" for turn in seg])
            output.append("")
        return '\n'.join(output)
    
    # ====================
    # Emotion-based segmentation (Steps 21-23)
    # ====================
    def _load_emotion_data(self, file_path: str) -> dict:
        """Load the JSON file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: Emotion data file '{file_path}' not found.")
            return {}

    def _enhance_emotion_tags(self, text: str) -> str:
        """Enhance the patient emotion expressions"""
        if self.emotion_data is None:
            self.emotion_data = self._load_emotion_data("Features of patient emotions.json")

        if not self.emotion_data:
            raise ValueError("Emotion data not loaded. Please check 'Features of patient emotions.json'.")

        emotion_prompt = "\n".join([
            f"{emotion}: {data['Features']} Example: {', '.join(data['Examples'])}"
            for emotion, data in self.emotion_data.items()
        ])

        enhancement_prompt = (
            "Based on the six basic emotions, please highlight the patient's emotions in each sentence and act as a patient. "
            "You can modify the sentence by adding punctuation marks (... / ! / ?), including interjections, or using parentheses for emotion annotations. "
            "Please do not change the original content of the conversation; only modify emotional expression.\n\n"
            f"{emotion_prompt}"
        )

        messages = [
            {"role": "system", "content": enhancement_prompt},
            {"role": "user", "content": text}
        ]

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=0.5
        )
        return response.choices[0].message.content

    def _segment_by_emotion(self, text: str) -> str:
        """Emotion-based"""
        cleaned_text = self.remove_timestamps(text)

        enhanced_text = self._enhance_emotion_tags(cleaned_text)

        segmentation_prompt = (
            "Segment medical conversation transcriptions based on the turning points in the patient's emotions. "
            "Ensure the medical conversation transcription is complete.\n\n"
            f"{enhanced_text}"
        )

        messages = [
            {"role": "system", "content": segmentation_prompt},
            {"role": "user", "content": cleaned_text}
        ]

        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
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
            selected_strategy = self._interactive_select_segmentation() 
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
        """Ask user if they want to continue or exit"""
        print("\nPress [Enter] to continue, [Esc] to exit and save progress...")
        while True:
            if keyboard.is_pressed("enter"):
                print("Continuing to next step...\n")
                return True
            elif keyboard.is_pressed("esc"):
                print("Exiting and saving current progress...\n")
                return False

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