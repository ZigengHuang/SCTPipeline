# SCTPipeline
SCTPipeline provides a default parameters version and a customized parameters (Steps 8-10) version for two Foundation LLM (ChatGPT-4o and Deepseek-R1).
According different Foundation LLM,you can choose the "SCTPipeline_ChatGPT-4o.py" or "SCTPipeline_Deepseek-R1.py".
* Default parameter version provides in the "SCTPipeline_ChatGPT-4o.py" or "SCTPipeline_Deepseek-R1.py"
* Customized parameter version provides in the "SCTPipeline_ChatGPT-4o_customized.py"or "SCTPipeline_Deepseek-R1_customized.py"

## Phase 1-1: Obtain RUCT by speech-to-text tools (Steps 1-3)
You use the "Phase 1-1_Whisper" file  to transcribe your recording audio to transcriptions based on whisper.
Optional, you can use the "Phase 1-1_Feishu" file to transcribe your recording audio to transcriptions based on Feishu.

## Phase 1-2,2,3: Pre-segment/transcription processing agent/prompts(Steps 4-23)
The section provided in the all default parameters version and a customized parameters (Steps 8-10) version files:

* Phase 1-2: Pre-segment RUCT into batches based on the number of characters;
* Phase 2: Developing transcription processing agent;
* Phase 3: Establishing prompts for SCT (Steps 4-23)

## Phase 4: Generation of SCT (Steps 24-28)
The "Generation_SCT.py" is provided for the Steps 24-28.

## Example: 
1.If you use default parameter version for Deepseek-R1:

`python SCTPipeline_Deepseek-R1.py -i /Example/RUCT1.txt -o /Output/SCT1.txt`

2. If you use customized version for Deepseek-R1:

`python SCTPipeline_Deepseek-R1.py -i /Example/RUCT1.txt /Example/manually_orgnized_RCT1.txt -o /Output/SCT2.txt`
![Workflow](https://github.com/user-attachments/assets/6b787e73-47bd-4dd4-9301-6ed1c503ff49)
![Alt Text](image.jpg){:width="400px"}



