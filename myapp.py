from transformers import AutoProcessor, AutoModelForCTC
from pydub import AudioSegment
# from fastapi import FastAPI
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
import os
import shutil
import soundfile as sf
import json
import librosa
import wave
import torch
from hindi import *
from telugu import *

app=FastAPI()

def func1(filepath,jsonfile,processor,model):
    wav_file = AudioSegment.from_file(file=filepath, format="wav")
    data = open(jsonfile)
    data = json.load(data)
    ans=""
    for j in range(len(data["filename"])):
        new = wav_file[data['filename'][j]['segment']['start']*1000 : data['filename'][j]['segment']['end']*1000]
        new.export(".venv\copie.wav", format="wav")
        y, sr = librosa.load(".venv\copie.wav")
        # data=sf.read(".venv\copie.wav")
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        if(new.duration_seconds>1):
            with torch.no_grad():
                logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                for k in transcription[0].split("<s>"):
                    ans+=k
                ans+=" "
    return ans

@app.post("/")
def text_generator(language,file1: UploadFile = File(...),file2: UploadFile = File(...)):
    with open(".venv\destination.wav", "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(".venv\destination.json", "wb") as buf:
        shutil.copyfileobj(file2.file, buf)
    if(os.path.exists(".venv\destination.wav")==True and os.path.exists(".venv\destination.json")==True):
        if(language=="hindi"):
            model , processor=hindi_model()
            ans=func1(".venv\destination.wav",".venv\destination.json",processor,model)
            return {"transcription":ans}
        elif(language=="telugu"):
            model , processor=telugu_model()
            ans=func1(".venv\destination.wav",".venv\destination.json",processor,model)
            return {"transcription":ans}
    else:
        return {"transcription":"Error"}