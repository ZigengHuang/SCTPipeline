# SCTPipeline: Agent-Based Clinical Transcription Processing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

SCTPipeline is an agent-based system for converting Raw Unstructured Clinical Transcripts (RUCT) into Structured Clinical Transcripts (SCT). The system employs a **Planner-Memory-Executor** cognitive architecture that autonomously coordinates four modular processing tools to handle heterogeneous clinical dialogue data without human supervision.

The Planner decomposes natural-language task instructions into dynamic tool chains. Memory maintains both short-term processing state and long-term domain knowledge. The Executor invokes tools through a safe-execution wrapper with automatic retry, validates outputs against tool contracts, and triggers structured repair sequences when inconsistencies arise.

## Architecture

```
                    +------------------+
                    |   Task Input     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |     Planner      |
                    | (LLM / Fallback) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |     Memory       |
                    | Short | Long     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Executor      |
                    | Invoke-Validate  |
                    | Repair-Replan    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     |   Tool 1   |  |   Tool 2    |  |   Tool 3    |
     |   Noise    |  |  Content    |  |  Speaker    |
     |  Removal   |  | Correction  |  |    ID       |
     +------------+  +-------------+  +-------------+
                             |
                    +--------v---------+
                    |     Tool 4       |
                    |  Segmentation    |
                    | Topic|Chrono|Emo |
                    +------------------+
                             |
                    +--------v---------+
                    |   SCT Output     |
                    +------------------+
```

## Features

### Four Processing Tools

1. **Noise Removal** -- Eliminates non-dialogue acoustic artifacts (hospital announcements, phone calls, environmental sounds) while preserving clinical content.
2. **Content Correction** -- Corrects ASR misrecognition errors, dialect expressions, near-homophone confusions, and enforces clinical safety checks (negation verification, dosage plausibility, temporal consistency).
3. **Speaker Identification** -- Assigns Doctor/Patient/Others role labels to each utterance based on content semantics and dialogue context.
4. **Segmentation** -- Organizes dialogue into coherent sections using one of three strategies:
   - **Topic-based**: Segments by clinical topic transitions using domain-specific templates.
   - **Chronological**: Segments by temporal gaps in timestamped dialogue.
   - **Emotion-based**: Segments by shifts in patient emotional state.

### Line-Preserving Contract

Tools 1-3 operate under a strict line-preserving contract: the number of output lines must exactly match the number of input lines. This invariant is enforced through chunk-level alignment and automatic repair when violated.

### Self-Correction

The Executor implements a three-stage repair sequence when validation fails:
1. Chunk-wise reprocessing to reduce input complexity
2. Re-invocation with identical parameters for transient failures
3. Deterministic rule-based fallback to generate a compliant output

## Installation

```bash
git clone https://github.com/ZigengHuang/SCTPipeline.git
cd SCTPipeline
pip install -r requirements.txt
```

## Quick Start

```bash
# Set your API credentials
export SCT_API_KEY="your-api-key"
export SCT_BASE_URL="https://api.openai.com/v1"

# Process a single file
python run.py -i sample_input/test_ruct.txt -o output/ \
    -t "denoise, correct and identify speaker"

# Batch process a folder with domain selection
python run.py -i input_folder/ -o output/ \
    -t "denoise, correct and identify speaker" --domain 2

# Use a specific model backbone
python run.py -i sample_input/test_ruct.txt -o output/ \
    -t "full pipeline" --model gpt-4o

# Resume interrupted batch processing
python run.py -i input_folder/ -o output/ \
    -t "denoise, correct and identify speaker" --resume
```

## Configuration

### API Settings

Set credentials via environment variables or command-line arguments:

| Method | API Key | Base URL |
|--------|---------|----------|
| Environment variable | `SCT_API_KEY` | `SCT_BASE_URL` |
| Command-line flag | `--api-key` | `--base-url` |

### Medical Domains

| Index | Domain |
|-------|--------|
| 1 | Health checkups |
| 2 | General outpatient visits |
| 3 | Surgical procedures |
| 4 | Hospitalization management |
| 5 | Customized domain-specific features |

### Knowledge Base

Each domain loads knowledge files from `knowledge/<domain_key>/`. Place your domain-specific `.txt` and `.pdf` files in the corresponding subdirectory. Each subdirectory must contain an `SCT.txt` reference file.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `gpt-4o` | LLM backbone model identifier |
| `--planner-model` | Same as `--model` | Model for the Planner module |
| `--domain` | `1` | Medical domain index (1-5) |
| `--chunk-size` | `50` | Lines per processing chunk |
| `--resume` | `false` | Skip already completed files in batch mode |

## Project Structure

```
SCTPipeline/
├── run.py                          # CLI entry point
├── agent.py                        # Planner, Memory, Executor modules
├── pipeline.py                     # Four processing tools + segmentation
├── config.py                       # Configuration and domain mapping
├── requirements.txt                # Python dependencies
├── prompts/
│   ├── domain_specific.json        # Domain-specific prompt templates
│   └── emotion_features.json       # Emotion feature definitions
├── knowledge/
│   └── clinical_corrections.txt    # Medical correction reference table
├── sample_input/
│   └── test_ruct.txt               # Sample RUCT input (synthetic data)
└── utils/
    ├── csv2txt.py                  # Convert CSV dialogue to TXT
    └── txt2csv.py                  # Merge processed TXT back to CSV
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
