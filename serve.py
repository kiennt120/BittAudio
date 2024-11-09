import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write as write_wav
import os
from fastapi import FastAPI, HTTPException, Body
import time
import torchaudio
import argparse
import uvicorn


parser = argparse.ArgumentParser()
parser.add_argument('--music_path', type=str, default="facebook/musicgen-large", help='Path to model file.')
parser.add_argument('--output_path', type=str, default="outputs", help='Path to save the generated music.')
parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate for the audio.")
parser.add_argument('--port', type=int, default=8000, help='Port to run the server.')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server.')
parser.add_argument('--debug', type=bool, default=False, help='Debug mode.')
args = parser.parse_args()

class MusicGenerator:
    def __init__(self, model_path="facebook/musicgen-medium"):
        """Initializes the MusicGenerator with a specified model path."""
        self.model_name = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def generate_music(self, prompt, token):
        """Generates music based on a given prompt and token count."""
        try:
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)
            audio_values = self.model.generate(**inputs, max_new_tokens=token)
            return audio_values[0, 0].cpu().numpy()
        except Exception as e:
            print(f"Error occurred with {self.model_name}: {e}")
            return None


def convert_music_to_tensor(audio_file):
        """Convert the audio file to a tensor."""
        try:
            _, file_extension = os.path.splitext(audio_file)
            if file_extension.lower() in ['.wav', '.mp3']:
                audio, sample_rate = torchaudio.load(audio_file)
                return audio[0].tolist()  # Convert to tensor/list
            else:
                
                return None
        except Exception as e:
            return None

        
app = FastAPI()
ttm_models = MusicGenerator(model_path=args.music_path)
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

@app.post("/generate_music")
def generate_music(text_input: str = Body(...), duration: int = Body(...)):
    start_time = time.time()
    print(f"text: {text_input}, duration: {duration}")
    path = os.path.join(output_path, text_input + ".wav")
    
    if os.path.isfile(path):
        print(f"Successfully generated '{text_input}'")
        print(f"time gen music: {time.time() - start_time}")
        return {"path": path}
    start_gen_music = time.time()
    music = ttm_models.generate_music(text_input, duration)
    print(f"time gen music: {time.time() - start_gen_music:.3f}")
    if music is None:
        print(f"Failed to generate music '{text_input}'")
        print(f"time: {time.time() - start_time}")
        return {"path": None}
    try:
        sample_rate = args.sample_rate
        write_wav(path, sample_rate, music)
        print(f"Successfully generated '{text_input}'")
        print(f"time: {time.time() - start_time}")
        return {"path": path}
    except Exception as e:
        print(f"Error occurred '{text_input}': {e}")
        print(f"time: {time.time() - start_time}")
        return {"path": None}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
    