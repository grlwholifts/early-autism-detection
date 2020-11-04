from fastapi import FastAPI, File, UploadFile
import uvicorn
from random import randint
import os

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # breakpoint()
    filename = str(randint(10000, 99999)) + ".wav"
    with open(filename, "ab") as f:
    	f.write(file.file.read())

    output_json = {"filename": file.filename}

    [Fs, x] = audioBasicIO.read_audio_file(filename)
    values, names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)

    for value, name in zip(values, names):
    	output_json[name] = ','.join(['%.7f' % num for num in value])

    # magic here

    print(f"Running -> rm {filename}")
    # os.system(f"rm {filename}")
    return output_json

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
