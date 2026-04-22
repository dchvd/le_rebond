import json
from xmlrpc import client

from faster_whisper import WhisperModel
from matplotlib import text
import openai

class TranscriberService:
    def __init__(self):
        client = openai.OpenAI(api_key="TON_API_KEY")

    def speech_to_text(self, audio_path):
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        full_text = ""
        for segment in segments:
            full_text += f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n"
        
        return full_text
    
    
    def text_to_database(self, text):
        prompt = f"""
        Analyse les commentaires de match de ping-pong suivants. 
        Extrais :
        1. Un résumé des moments clés.
        2. Liste des points positifs et négatifs.
        3. Une note de performance globale sur 10.
        Réponds EXCLUSIVEMENT au format JSON.

        Texte : {text[:4000]} # Limitation de contexte simplifiée
        """
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)