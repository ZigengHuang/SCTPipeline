import whisper
import os

def transcribe_with_whisper(audio_file_path):

    model = whisper.load_model("base")  #

    result = model.transcribe(audio_file_path)

    transcription = []
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        speaker = "speaker1" 
        transcription.append(f"[{start_time:.2f}-{end_time:.2f}] {speaker}: {text}")


    save_to_txt(transcription, "whisper_transcription_result.txt")

def save_to_txt(transcription, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for line in transcription:
            file.write(line + '\n')
    print(f"Transcription result saved to {file_name}")

if __name__ == '__main__':
    audio_file_path = "test.mp3"  
    transcribe_with_whisper(audio_file_path)
