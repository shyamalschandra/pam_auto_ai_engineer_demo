import torch
import whisper
import pyaudio
import wave
import sys
import os
import re
import asyncio
import ollama
import tkinter as tk
from tkinter import ttk
from threading import Thread, Lock
from queue import Queue, Empty
import time
import numpy as np
import warnings
import subprocess

warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

result_outer = ""
speak_outer = ""
recording = False
listening = False
audio_queue = Queue(maxsize=1)
STOP_WORD = "roger"  # Change this to your preferred stop word
lock = Lock()

# Create the root window first
root = tk.Tk()
root.title("Voice Assistant")
root.geometry("400x600")
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

# Add these global variables
input_devices = []
output_voices = []
selected_input_device = tk.StringVar()
selected_output_voice = tk.StringVar()

# Add these functions to get input devices and output voices
def get_input_devices():
    global input_devices
    p = pyaudio.PyAudio()
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            input_devices.append(dev['name'])
    p.terminate()
    return input_devices

def get_output_voices():
    global output_voices
    result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
    output_voices = [line.split()[0] for line in result.stdout.split('\n') if line]
    return output_voices

def record_audio():
    global recording
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    audio = pyaudio.PyAudio()

    # Get the index of the selected input device
    device_index = None
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if dev['name'] == selected_input_device.get():
            device_index = i
            break

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
                        frames_per_buffer=CHUNK, input_device_index=device_index)
    print("Recording started...")

    frames = []
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Write to a temporary file for real-time transcription
        wf = wave.open("temp_audio.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_queue.put("temp_audio.wav")

def transcribe_audio(audio):
    model = whisper.load_model("base")  # Change "tiny" to "base" or "small"
    result = model.transcribe(audio, language="en")
    return result["text"].lower()

def continuous_recognition():
    global recording, listening, result_outer
    model = whisper.load_model("tiny")
    
    while listening:
        with lock:
            recording = True
        Thread(target=record_audio).start()

        accumulated_text = ""
        start_time = time.time()
        
        while recording:
            if os.path.exists("temp_audio.wav"):
                chunk_text = transcribe_audio("temp_audio.wav")
                if chunk_text != accumulated_text:
                    new_text = chunk_text[len(accumulated_text):].strip()
                    accumulated_text = chunk_text
                    result_outer = accumulated_text
                    root.after(0, update_result_label)

                    print(f"Recognized: {new_text}")  # Debug print

                    if STOP_WORD in new_text:
                        with lock:
                            recording = False
                        break

            time.sleep(0.5)  # Adjust this value to balance between responsiveness and CPU usage

        end_time = time.time()
        elapsed_time = end_time - start_time
        root.after(0, lambda: update_task_time('speech_recog', elapsed_time))
        root.after(0, lambda: progress_vars['speech_recog'].set(100))

        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

        llm_insert()

async def speak(voice):
    global speak_outer, listening
    message = str(speak_outer)
    root.after(0, update_speak_label)

    with lock:
        listening = False
    
    # Set progress bar to 100% before speech synthesis
    root.after(0, lambda: progress_vars['speech_synthesis'].set(100))
    
    # Calculate and update the task time before speech synthesis
    start_time = time.time()
    estimated_duration = len(message) * 0.1  # Rough estimate: 0.1 seconds per character
    root.after(0, lambda: update_task_time('speech_synthesis', estimated_duration))
    
    # Play the audio with the selected voice
    os.system(f"say -v \"{selected_output_voice.get()}\" -r 150 \"{message}\"")
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Update the task time with the actual duration
    root.after(0, lambda: update_task_time('speech_synthesis', actual_duration))

    # Zero out all stopwatches and progress bars
    root.after(0, zero_out_ui)

    with lock:
        listening = True
    root.after(0, resume_listening)

def zero_out_ui():
    for var in progress_vars.values():
        var.set(0)
    
    for var in task_times.values():
        var.set("00:00.00")
    
    update_result_label()
    update_speak_label()

def llm_insert():
    global result_outer, speak_outer
    start_time = time.time()
    try:
        q = result_outer
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
    
    asyncio.run(speak("Zoe (Premium)"))

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

def start_listening():
    global listening, result_outer, speak_outer
    if not listening:
        listening = True
        result_outer = ""
        speak_outer = ""
        
        for var in progress_vars.values():
            var.set(0)
        
        for var in task_times.values():
            var.set("00:00.00")
        
        update_result_label()
        update_speak_label()
        
        Thread(target=continuous_recognition).start()

def resume_listening():
    global listening
    if listening:
        Thread(target=continuous_recognition).start()

def stop_listening():
    global listening, recording
    with lock:
        listening = False
        recording = False

def on_start_button_click():
    start_listening()
    record_button.config(text="Stop Listening", command=on_stop_button_click)

def on_stop_button_click():
    stop_listening()
    record_button.config(text="Start Listening", command=on_start_button_click)

def update_task_time(task, elapsed_time):
    minutes, seconds = divmod(int(elapsed_time), 60)
    centiseconds = int((elapsed_time - int(elapsed_time)) * 100)
    task_times[task].set(f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}")

# Add these functions to create the input and output selection widgets
def create_input_selection():
    frame = ttk.Frame(root)
    frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(frame, text="Input Device:").pack(side=tk.LEFT)
    input_combo = ttk.Combobox(frame, textvariable=selected_input_device, values=get_input_devices())
    input_combo.pack(side=tk.LEFT, padx=(5, 0))
    input_combo.set(input_devices[0] if input_devices else "")

def create_output_selection():
    frame = ttk.Frame(root)
    frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(frame, text="Output Voice:").pack(side=tk.LEFT)
    output_combo = ttk.Combobox(frame, textvariable=selected_output_voice, values=get_output_voices())
    output_combo.pack(side=tk.LEFT, padx=(5, 0))
    output_combo.set(output_voices[0] if output_voices else "")

# Modify your main UI creation to include the new widgets
def create_ui():
    global record_button, result_text, speak_text

    create_input_selection()
    create_output_selection()

    record_button = ttk.Button(root, text="Start Listening", command=on_start_button_click)
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
        
        ttk.Label(frame, text="%").pack(side=tk.LEFT)

    # Add timers at the bottom
    timer_frame = ttk.Frame(root)
    timer_frame.pack(fill=tk.X, padx=10, pady=10)

    for task, var in task_times.items():
        task_frame = ttk.Frame(timer_frame)
        task_frame.pack(side=tk.LEFT, expand=True)
        
        ttk.Label(task_frame, text=f"{task.replace('_', ' ').title()}:").pack()
        ttk.Label(task_frame, textvariable=var, font=("Courier", 12)).pack()

# Call create_ui() instead of creating UI elements directly
create_ui()

root.mainloop()