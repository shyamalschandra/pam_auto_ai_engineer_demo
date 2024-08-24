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
import tkinter as tk
from tkinter import ttk
from threading import Thread
from queue import Queue, Empty
import time

result_outer = ""
speak_outer = ""
recording = False
audio_queue = Queue(maxsize=1)  # Queue to hold the most recent audio file

# Create the root window first
root = tk.Tk()
root.title("Voice Recorder")
root.geometry("400x600")  # Increased height to accommodate timers
root.resizable(True, True)

# Now create the Tkinter variables
progress_vars = {
    'speech_recog': tk.DoubleVar(root),
    'llm_inference': tk.DoubleVar(root),
    'speech_synthesis': tk.DoubleVar(root)
}

task_times = {
    'speech_recog': tk.StringVar(root, value="00:00.00"),
    'llm_inference': tk.StringVar(root, value="00:00.00"),
    'speech_synthesis': tk.StringVar(root, value="00:00.00")
}

def record():
    global recording
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

    # Clear the buffer by reading and discarding any existing data
    stream.read(stream.get_read_available())

    while recording:
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

    # Put the filename in the queue, replacing any existing item
    if not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except Empty:
            pass
    audio_queue.put(WAVE_OUTPUT_FILENAME)

async def speak(voice):
    global speak_outer
    message = str(speak_outer)
    root.after(0, update_speak_label)  # Update the label before speaking
    
    # Simulate speech synthesis steps
    steps = [
        ("Text preprocessing", 10),
        ("Phoneme generation", 30),
        ("Audio waveform synthesis", 60)
    ]
    
    progress = 0
    start_time = time.time()
    for step, step_progress in steps:
        for i in range(step_progress):
            time.sleep(0.01)  # Adjust sleep time based on actual processing time
            progress += 1
            root.after(0, lambda p=progress: progress_vars['speech_synthesis'].set(p))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    root.after(0, lambda: update_task_time('speech_synthesis', elapsed_time))
    
    os.system("say -v \"" + str(voice) + "\" -r 150 \""+ str(message) +"\"")
    root.after(0, lambda: progress_vars['speech_synthesis'].set(100))

async def convert_from_wav_to_mp3(filename):
    os.system(f"ffmpeg -y -v quiet -i {filename} -ar 16000 -ac 1 -c:a pcm_s16le cleanFile.wav")

async def speech_recog(filename):
    global result_outer
    start_time = time.time()
    try:
        model = whisper.load_model("tiny")
        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Update progress for model loading and audio preprocessing
        for i in range(30):
            time.sleep(0.01)
            root.after(0, lambda: progress_vars['speech_recog'].set(i))
        
        _, probs = model.detect_language(mel)
        options = whisper.DecodingOptions(fp16=False)
        
        # Update progress for language detection
        for i in range(30, 50):
            time.sleep(0.01)
            root.after(0, lambda: progress_vars['speech_recog'].set(i))
        
        result = whisper.decode(model, mel, options)
        
        # Update progress for decoding
        for i in range(50, 100):
            time.sleep(0.01)
            root.after(0, lambda: progress_vars['speech_recog'].set(i))
        
        result_no_dots = re.sub(r'\.', '', result.text)
        result_no_dots = result_no_dots.lower()
        print(result_no_dots)
        result_outer = result_no_dots
        root.after(0, update_result_label)
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        result_outer = "Error during speech recognition"
        root.after(0, update_result_label)
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        root.after(0, lambda: update_task_time('speech_recog', elapsed_time))
        root.after(0, lambda: progress_vars['speech_recog'].set(100))

def llm_insert():
    global result_outer
    global speak_outer
    start_time = time.time()
    try:
        q = result_outer
        
        # Simulate LLM processing steps
        steps = [
            ("Tokenizing input", 20),
            ("Processing through layers", 60),
            ("Generating output", 20)
        ]
        
        progress = 0
        for step, step_progress in steps:
            for i in range(step_progress):
                time.sleep(0.02)  # Adjust sleep time based on actual processing time
                progress += 1
                root.after(0, lambda p=progress: progress_vars['llm_inference'].set(p))
        
        response = ollama.chat(model='tinyllama', messages=[{'role': 'user','content': q,},])
        r1 = response['message']['content']
        speak_outer = r1
        print("Prompt reply: " + r1 + "")
    except Exception as e:
        print(f"Error during LLM insertion: {e}")
        speak_outer = "Error during LLM insertion"
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        root.after(0, lambda: update_task_time('llm_inference', elapsed_time))
        root.after(0, lambda: progress_vars['llm_inference'].set(100))

async def process_audio(filename):
    await convert_from_wav_to_mp3(filename)
    await speech_recog('cleanFile.wav')
    llm_insert()
    await speak("Zoe (Premium)")

def audio_processor():
    while True:
        try:
            filename = audio_queue.get(timeout=1)
            asyncio.run(process_audio(filename))
        except Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio processor: {e}")

def update_result_label():
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Recognized: {result_outer}")
    result_text.config(state=tk.DISABLED)

def update_speak_label():
    speak_text.config(state=tk.NORMAL)
    speak_text.delete(1.0, tk.END)
    speak_text.insert(tk.END, f"Synthesized: {speak_outer}")
    speak_text.config(state=tk.DISABLED)

def start_recording():
    global recording, result_outer, speak_outer
    if not recording:
        result_outer = ""
        speak_outer = ""
        recording = True
        print("recording started")
        
        # Reset progress bars
        for var in progress_vars.values():
            var.set(0)
        
        # Reset stopwatch times
        for var in task_times.values():
            var.set("00:00.00")
        
        # Update the UI labels
        update_result_label()
        update_speak_label()
        
        Thread(target=record).start()

def stop_recording():
    global recording
    if recording:
        recording = False
        print("recording stopped")

def on_button_press(event):
    start_recording()

def on_button_release(event):
    stop_recording()

def update_task_time(task, elapsed_time):
    minutes, seconds = divmod(int(elapsed_time), 60)
    centiseconds = int((elapsed_time - int(elapsed_time)) * 100)
    task_times[task].set(f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}")

# Create and pack UI elements
record_button = ttk.Button(root, text="Hold to Record")
record_button.pack(pady=20)

result_text = tk.Text(root, height=5, wrap=tk.WORD)
result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
result_text.insert(tk.END, "Recognized: ")
result_text.config(state=tk.DISABLED)

speak_text = tk.Text(root, height=5, wrap=tk.WORD)
speak_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
speak_text.insert(tk.END, "Synthesized: ")
speak_text.config(state=tk.DISABLED)

# Add progress bars with labels
for task, var in progress_vars.items():
    frame = ttk.Frame(root)
    frame.pack(fill=tk.X, padx=10, pady=5)
    
    label = ttk.Label(frame, text=f"{task.replace('_', ' ').title()}:")
    label.pack(side=tk.LEFT)
    
    progress_bar = ttk.Progressbar(frame, variable=var, maximum=100, length=200, mode='determinate')
    progress_bar.pack(side=tk.LEFT, padx=(5, 0))
    
    percentage_label = ttk.Label(frame, textvariable=var)
    percentage_label.pack(side=tk.LEFT, padx=(5, 0))
    
    # Add % sign
    ttk.Label(frame, text="%").pack(side=tk.LEFT)

# Add timers at the bottom
timer_frame = ttk.Frame(root)
timer_frame.pack(fill=tk.X, padx=10, pady=10)

for task, var in task_times.items():
    task_frame = ttk.Frame(timer_frame)
    task_frame.pack(side=tk.LEFT, expand=True)
    
    ttk.Label(task_frame, text=f"{task.replace('_', ' ').title()}:").pack()
    ttk.Label(task_frame, textvariable=var, font=("Courier", 12)).pack()

record_button.bind('<ButtonPress-1>', on_button_press)
record_button.bind('<ButtonRelease-1>', on_button_release)

# Start the audio processor thread
Thread(target=audio_processor, daemon=True).start()

root.mainloop()