# SCTPipeline
SCTPipeline provides a comprehensive default and custom parameter (Steps 8-10) version for both Foundation LLMs (ChatGPT-4o and Deepseek-R1).
According different Foundation LLMs, you can choose the "SCTPipeline_ChatGPT-4o.py" or "SCTPipeline_Deepseek-R1.py".
* Run this pipeline, you can choose 1-4 option (Health checkups, Routine outpatient guidance, Surgery, and Hospitalization guidance) to prefined prompts according to domain-specific features (Default). The pipeline provides four medical knowlegde for the 1-4 options in './medical_knowledge_backup' folder. 
  
* If you want to customized medical domain features (Phase 2, Steps 8-10), you can choose option 5_Customized domain-specific features. You need to ensure SCT manually organized add in the './medical_knowledge_backup/option 5_General medical knowledge' folder.

## Phase 1-1: Obtain RUCT by speech-to-text tools (Steps 1-3)
* Recommend：You use the "Phase 1-1_Whisper" file  to transcribe your recording audio to obtain RUCT based on whisper.
* Optional, you can use the Feishu to transcribe your recording audio to obtain RUCT based on Feishu web.

## Phase 1-2,2,3,4: Pre-segment/transcription processing agent/prompts/Generation of SCT(Steps 4-28)

* Phase 1-2: Pre-segment RUCT into batches based on the number of characters (Steps 4-6);
* Phase 2: Developing transcription processing agent (Steps 7-13);
  In this phase, you need to ensure './medical_knowledge/' folder include Department medical knowledge datasets (The SCT.txt for SCTPipleline or your customized SCT.txt), General medical knowledge datasets,and Dialect dictionaries (optional).
* Phase 3: Establishing prompts for SCT (Steps 14-23)
* Phase 4: Generation of SCT (Steps 24-28)

## Usage Example: 
Take 'Deepseek-R1','option 3','./Example/RUCT.txt' as an example：

`python SCTPipeline_Deepseek-R1.py -i ./Example/RUCT.txt -o /Output/processed_sct_example.txt`
<img src="https://github.com/user-attachments/assets/6b787e73-47bd-4dd4-9301-6ed1c503ff49" style="width:60%; height:auto;" alt="Workflow" />





