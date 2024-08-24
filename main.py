# Author: Shyamal Suhana Chandra
# Date: August 16, 2024
# Project: Magical Toys, Level 1-4
# Due Date: August 18, 2024

# Boilerplate code for PyAudio taken from: https://realpython.com/playing-and-recording-sound-python/#pyaudio

import torch
import whisper
import pyaudio
import wave
import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pyffmpeg import FFmpeg
import re
import asyncio
import ollama
import pygame

from TTS.api import TTS

result_outer = ""
speak_outer = ""

def record():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    WAVE_OUTPUT_FILENAME = "recordedFile.wav"
    device_index = 2
    audio = pyaudio.PyAudio()

    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    index = int(0)
    print("recording via index " + str(index))

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=index, frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []

    recording = True
    while recording:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r:  # Stop recording on key release
                    recording = False
                    break
        if recording:
            data = stream.read(CHUNK)
            Recordframes.append(data)

    print("recording stopped")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
    return asyncio.sleep(0)

def speak(voice):

    global speak_outer

    message = str(speak_outer)

    os.system("say -v \"" + str(voice) + "\" -r 150 \""+ str(message) +"\"")
    return asyncio.sleep(0)

def convert_from_wav_to_mp3():

    os.system("ffmpeg -y -v quiet -i recordedFile.wav -ar 16000 -ac 1 -c:a pcm_s16le cleanFile.wav")
    return asyncio.sleep(0)

def speech_syn():

    global speak_outer

    os.environ["COQUI_TOS_AGREED"] = "1"

    device = "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(text=str(speak_outer), speaker_wav="shyamal.wav", language="en", file_path="output.wav")

def speech_recog(filename):

        global result_outer

        # load audio and pad/trim it to fit 30 seconds
        model = whisper.load_model("tiny")
        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        #print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)

        result_no_dots = re.sub(r'\.', '', result.text)
        result_no_dots = result_no_dots.lower()

        # print the recognized text
        print(result_no_dots)
        result_outer = result_no_dots
        return asyncio.sleep(0)

def llm_insert():
    global result_outer
    global speak_outer

    # Boilerplate code taken from: https://apmonitor.com/dde/index.php/Main/LargeLanguageModel
    q = result_outer
    response = ollama.chat(model='openhermes2.5-mistral', messages=[{'role': 'user','content': q,},])
    r1 = response['message']['content']
    speak_outer = r1
    print("Prompt reply: " + r1 + "")

async def main():
    global result_outer
    global speak_outer

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Press 'R' to start/stop recording")
        
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Start recording on key press
                    print("Recording voice!")
                    await record()
                    print("Converting wav to mp3!")
                    await convert_from_wav_to_mp3()
                    print("Speech recognition with Whisper!")
                    await speech_recog('cleanFile.wav')
                    print("Chatting with the LLM!")
                    llm_insert()
                    print("Speak the response based on the chat interface with LLM!")
                    await speak("Zoe (Premium)")
                    print("Speak with Tortoise TTS")
                    # speech_syn()
                    # os.system(f"mplayer output.wav")
                    print("Demo Done")

if __name__ == "__main__":
    asyncio.run(main())