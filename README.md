# SCTPipeline
SCTPipeline provides a comprehensive default and custom parameter (Steps 8-10) version for both Foundation LLMS (ChatPGT-40 and Deepseek-R1).
According different Foundation LLM,you can choose the "SCTPipeline_ChatGPT-4o.py" or "SCTPipeline_Deepseek-R1.py".
* Run this pipeline, you need to choose 1-4 option to prefined prompts according to domain-specific features (Default).
* If you want to customized medical domain features (Phase 2, Steps 8-10), you can choose option 5:Customized domain-specific features.

## Phase 1-1: Obtain RUCT by speech-to-text tools (Steps 1-3)
* Recommend：You use the "Phase 1-1_Whisper" file  to transcribe your recording audio to transcriptions based on whisper.
* Optional, you can use the "Phase 1-1_Feishu" file to transcribe your recording audio to transcriptions based on Feishu.

## Phase 1-2,2,3: Pre-segment/transcription processing agent/prompts(Steps 4-23)

* Phase 1-2: Pre-segment RUCT into batches based on the number of characters (Steps 4-6);
* Phase 2: Developing transcription processing agent (Steps 7-13);
  In this phase, you need to ensure ./medical/ folder include Department medical knowledge datasets (The default SCT for SCTPipleline or your manually organized SCT), General medical knowledge datasets,and Dialect dictionaries (optional).
* Phase 3: Establishing prompts for SCT (Steps 14-23)

## Phase 4: Generation of SCT (Steps 24-28)
The "Generation_SCT.py" is provided for the Steps 24-28 to generate SCT.

## Usage Example: 
Take Deepseek-R1 as an example：

1.If you use default parameter version (option 1-4) for Deepseek-R1:

`python SCTPipeline_Deepseek-R1.py -i /Example/RUCT1.txt -k /medical/ -o /Output/SCT1.txt`

2. If you use customized version for Deepseek-R1:

`python SCTPipeline_Deepseek-R1.py -i /Example/RUCT1.txt /Example/manually_orgnized_RCT1.txt -k /medical.txt -o /Output/SCT2.txt`
<img src="https://github.com/user-attachments/assets/6b787e73-47bd-4dd4-9301-6ed1c503ff49" style="width:60%; height:auto;" alt="Workflow" />





