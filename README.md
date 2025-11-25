# Medical Transcription Agent: From Raw to Structured Clinical Transcripts

A novel agent-based framework for transforming Raw Unstructured Clinical Transcripts (RUCT) into Structured Clinical Transcripts (SCT) through autonomous planning, memory-augmented processing, and modular tool execution.

## Overview

This project implements an intelligent agent-based system for processing medical conversation transcriptions, as detailed in the associated research. The framework systematically converts RUCT into SCT using a cognitive core (Planner-Memory-Executor) architecture combined with specialized modular tools for:

- Noise removal and data cleaning
- Medical content correction and standardization
- Speaker identification and role attribution
- Semantic and temporal segmentation

## Key Features

### 🗣️ Multilingual Clinical Support
- **Primary Languages**: Mandarin, Wuhan dialect, Cantonese
- **Extended Support**: English clinical dialogues
- **Dialect Adaptation**: Specialized processing for regional Chinese variations

### 🛡️ Robust Processing Framework
- Deterministic planning with fallback mechanisms
- Line-preserving execution contracts for timestamp integrity
- Comprehensive validation and repair pipelines
- Domain-adaptive memory system for clinical specialty customization

### 📊 Validated Performance
- Tested on 7,197 minutes of real-world outpatient conversations
- 780 cases across 8 clinical departments
- Multi-hospital validation (Wuhan and Shenzhen)

## Dataset Characteristics

### Primary Corpus
| Metric | Specification |
|--------|---------------|
| Duration | 7,197 minutes |
| Cases | 780 patients |
| Departments | 8 specialties |
| Languages | Mandarin, Wuhan dialect, Cantonese |

### Extended Validation
- **English Corpus**: 78 dialogues (240 minutes)
- **Sources**: Standardized patient simulations, educational materials, medical dramas
- **Privacy**: Full HIPAA compliance with 18 PHI categories de-identified

## Technical Architecture

### Data Preprocessing Pipeline
1. **ASR Transcription**
   - Whisper (multilingual offline processing)
   - Feishu Minutes (Chinese dialect optimization)

2. **Privacy Protection**
   - HIPAA-compliant de-identification
   - Protected health information removal

3. **Processing Optimization**
   - 2,000–3,000 character chunking for LLM efficiency
   - Comprehensive metadata logging for reproducibility

### Core Agent Framework

#### Planner Component
- Generates structured execution plans from task instructions
- Deterministic configuration (seed=42, temperature=0)
- Fallback mechanism for LLM dependency resilience
- Stateless design with JSON-formatted output

#### Memory System
- **Short-term**: Transient results and intermediate outputs
- **Long-term**: Clinical knowledge, prompt templates, domain resources
- Atomic read/write operations with UTF-8 encoding
- Consistent checkpointing via ordered key dumping

#### Executor Engine
- Tool validation and safe invocation with retry logic
- Post-execution validation against tool contracts
- State-preserving repair sequences and replanning
- Comprehensive execution tracing and logging

### Modular Tool Ecosystem

| Tool | Purpose | Key Capabilities |
|------|---------|------------------|
| Noise Removal | Eliminate non-clinical artifacts | Background noise filtering, line preservation |
| Content Correction | Ensure medical accuracy | Terminology standardization, dialect adaptation |
| Speaker Identification | Role attribution | Lexical/pragmatic cue analysis |
| Segmentation | Dialogue organization | Topical, chronological, emotional structuring |

### Execution Contracts
- Strict line-count preservation for timestamp alignment
- Chunk-wrapper enforcement with explicit prompt rules
- Cross-module consistency maintenance
- Automated post-processing line restoration

## Quality Assurance

### Validation Protocols
- Line-count agreement verification
- Speaker label accuracy assessment
- Non-empty segmentation validation
- Contextual continuity checks

### Repair Mechanisms
- Chunk-wise reprocessing pipelines
- Simplified argument retry logic
- Rule-based fallback strategies
- End-to-end batch processing wrappers

## SCT Assembly & Output

### Multi-layer Integration
- **Semantic Layer**: Medical terminology and factual consistency
- **Temporal Layer**: Chronological organization and timing alignment
- **Emotional Layer**: Affective state tracking and annotation

### Export Formats
- JSON (structured data interchange)
- CSV (tabular analysis and processing)
- Plain text (human-readable documentation)

## Domain Adaptation System

### Memory Architecture
- Structured dictionaries for clinical knowledge
- Localized medical patterns and SCT fragments
- Versioned prompt template management

### Intelligent Retrieval
- Metadata extraction from transcripts
- Priority-based domain entry access
- Context merging before tool invocation
- Reproducible state reconstruction via hashing

### Domain Expansion
- Guided extraction for new clinical specialties
- Lexical cue generation from transcript summaries
- Append-only update strategy for version control

## Performance Evaluation

### Comprehensive Metrics

| Task | Evaluation Metrics | Methodology |
|------|-------------------|-------------|
| Noise Removal | Noise removal rate, noise ratio | Human-annotated categories (3-annotator agreement) |
| Content Correction | Cosine similarity to reference SCT | Annotator-verified transcript comparison |
| Speaker Identification | Utterance-level accuracy | Consensus labeling (Cohen's κ ≥ 0.8) |
| Segmentation | Topic/chronology/emotion accuracy | Utterance-level human annotation alignment |

## Experimental Validation

### Noise Perturbation Analysis
- Gold-standard SCT corpus (780 Chinese segments)
- Controlled noise injection at 10%-100% increments
- Corruption strategies: contextual disturbances, homophonic substitutions
- Evaluation metric: BERTScore (F1) against clean references

### Tool Ablation Study
- **Configurations Tested**: Full agent baseline vs. individual tool ablation
- **Test Set**: 50 held-out dialogues (department/topic/emotion balanced)
- **Performance Metrics**: Correction accuracy, speaker accuracy, segmentation accuracy, RAD

## Implementation Specifications

### Hardware Requirements
- **Standard Workstations**: Windows 11 (Intel i7-13700HX + RTX 4080)
- **Apple Systems**: macOS with Apple M1 chip or later

### Software Environment
- **Python**: 3.11
- **IDE**: PyCharm 2025.2.2
- **Key Libraries**: 
  - Hugging Face Transformers (v4.35.0)
  - sklearn_crfsuite
  - SentencePiece

### Model Dependencies
- **Cognitive Core**: ChatGPT-4o
- **ASR Processing**: Whisper/Feishu Minutes
- **Embedding Generation**: SentenceTransformer (all-MiniLM-L6-v2)

## Usage Instructions

### Basic Command
```bash
python genuine_agent.py \
  --input /path/to/input \
  --output /path/to/output_dir \
  --task "Process clinical transcript" \
  [--planner-model gpt-4o] \
  [--domain 1-5] \
  [--resume]

```
### Parameter Reference
| Parameter | Description | Default |
|------|-------------------|-------------|
| --input, -i | Input RUCT file/folder (.txt) | Required|
| --output, -o | Output directory for SCT and logs | Required |
| --task, -t | Processing task description | Required |
| --planner-model | LLM for planning tasks | gpt-4o |
| --domain | Clinical specialty index (1-5) | Optional |
| --resume | Resume batch processing| Flag |
		

## Project Structure
```text
medical-transcription-agent/
├── genuine_agent.py                 # CLI entry point & batch processing
├── assistant_agent.py               # Core agent implementation
├── medical_transcription_pipeline.py # Main processing pipeline
├── Domain Specific Prompts.json     # Clinical domain resources
├── Features of patient emotions.json # Emotion segmentation data
├── prompt指南.txt                   # Tool prompt templates
├── Data_processing/                 # Utility scripts
│   ├── csv2txt.py
│   └── additional_utilities
└── README.md                        # Project documentation
```

## License
This project is licensed under the MIT License. See LICENSE file for complete details.

## Acknowledgments
### Data Sources
- Real-world clinical conversations from hospitals in Wuhan and Shenzhen, China
- English dialogue resources from Geeky Medics, Stanford Medicine, BMJ Learning
- Medical drama transcripts for supplementary validation

## Citation
If using this work in academic research, please cite:
~~~
@article{name,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
~~~

## Contact
For research collaborations or technical inquiries, please feel free to contact us for any questions or comments: 
- Hao Qin, E-mail: qinhao237@gmail.com; 
- Peixing Wan, E-mail: peixing@bjmu.edu.cn;
- Erping Long, E-mail: erping.long@ibms.pumc.edu.cn.
