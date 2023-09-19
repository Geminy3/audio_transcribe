import whisper
import torch
import os
import time
import pandas as pd


#if os.path.exists("test.xlsx"):
#    tab = pd.read_excel("test.xlsx")
#else:
#    tab = pd.DataFrame()

### Write the test tab

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)




model = whisper.load_model("medium").to(DEVICE)
audio_path = os.path.join("audio.mp3")

language = "fr"

print("launch transcription")
start = time.time()
result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps = True
    )

len_time = time.time() - start
print(len_time)
for bloc in result["segments"]:
    print(bloc["start"], bloc["end"], bloc["text"])

with open("out.txt", "w", encoding = "utf-8") as f:
    txt = ""
    for bloc in result["segments"]:
        txt += str(bloc["start"]) + "-" + str(bloc["end"]) + " : " + bloc["text"] + "\n"
    f.write(txt)
    


