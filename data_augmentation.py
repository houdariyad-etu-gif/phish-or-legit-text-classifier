"""
data_augmentation.py
Génération de données synthétiques pour équilibrer le dataset (Data Augmentation).
Utilise un petit LLM pour générer de nouveaux messages de phishing.
"""
from transformers import pipeline
import pandas as pd

def generate_phishing_variations(text, num_variations=3):
    print("Chargement du modèle de génération (ex: GPT-2 ou T5)...")
    # On utilise un modèle Hugging Face pour générer des variations
    generator = pipeline("text-generation", model="distilgpt2")
    
    prompt = f"Write a realistic phishing SMS similar to: '{text}'\nPhishing SMS:"
    
    variations = []
    for _ in range(num_variations):
        result = generator(prompt, max_new_tokens=30, num_return_sequences=1)
        # Nettoyage du texte généré
        new_text = result[0]['generated_text'].split("Phishing SMS:")[-1].strip()
        variations.append(new_text)
    
    return variations

if __name__ == "__main__":
    # Exemple : on a peu de phishing (20%), on en génère pour équilibrer
    sample_phish = "Your account is locked. Verify now http://bit.ly/abc"
    new_data = generate_phishing_variations(sample_phish, num_variations=5)
    
    print("Nouveaux messages générés :")
    for msg in new_data:
        print(f" - {msg}")