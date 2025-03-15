# SCTPipeline
SCTPipeline provides a comprehensive default and custom parameter (Steps 8-10) version for both Foundation LLMs (ChatPGT-40 and Deepseek-R1).
According different Foundation LLMs, you can choose the "SCTPipeline_ChatGPT-4o.py" or "SCTPipeline_Deepseek-R1.py".
* Run this pipeline, you can choose 1-4 option (health checkups, general outpatient visits, surgical procedures, and hospitalization management) to prefined prompts according to domain-specific features (Default).The pipeline provides four medical knowlegde for the 1-4 options. According to your option, you need to change the 'medical_knowledge' to make the pipeline work well. Critical:Replace the folder name with 'medical_knowledge'.
  
* If you want to customized medical domain features (Phase 2, Steps 8-10), you can choose option 5:Customized domain-specific features.You need to ensure SCT manually organized add in the 'General medical knowledge' folder and replace the folder name with 'medical_knowledge'.

We provide an example of surgery to run the pipeline.'medical knowledge' folder contain surgery knowledge files to support processing the RUCT file in the 'Example' folder.

## Phase 1-1: Obtain RUCT by speech-to-text tools (Steps 1-3)
* Recommend：You use the "Phase 1-1_Whisper" file  to transcribe your recording audio to obtain RUCT based on whisper.
* Optional, you can use the "Phase 1-1_Feishu" file to transcribe your recording audio to obtain RUCT based on Feishu.

## Phase 1-2,2,3,4: Pre-segment/transcription processing agent/prompts/Generation of SCT(Steps 4-28)

* Phase 1-2: Pre-segment RUCT into batches based on the number of characters (Steps 4-6);
* Phase 2: Developing transcription processing agent (Steps 7-13);
  In this phase, you need to ensure ./medical_knowledge/ folder include Department medical knowledge datasets (The SCT.txt for SCTPipleline or your customized SCT.txt), General medical knowledge datasets,and Dialect dictionaries (optional).
* Phase 3: Establishing prompts for SCT (Steps 14-23)
* Phase 4: Generation of SCT (Steps 24-28)
T
## Usage Example: 
Take Deepseek-R1 as an example：

1.If you use default parameter version (option 1-4) for Deepseek-R1:

`python SCTPipeline_Deepseek-R1.py -i /Example/RUCT1.txt -k /medical/ -o /Output/processed_sct_example.txt`

2. If you use customized version for Deepseek-R1:

`python SCTPipeline_Deepseek-R1.py -i ./Example/RUCT.txt ./medical_knowledge/SCT.txt" -k /medical.txt -o /Output/processed_sct_example.txt`
<img src="https://github.com/user-attachments/assets/6b787e73-47bd-4dd4-9301-6ed1c503ff49" style="width:60%; height:auto;" alt="Workflow" />





