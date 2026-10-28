
"""
data_collection.py
Script de collecte de données réelles (Hugging Face / Kaggle).
Ce script montre comment passer du dataset synthétique à un dataset réel.
Exécution (optionnelle) : python data_collection.py
"""
from datasets import load_dataset
import pandas as pd

def get_huggingface_data():
    print("1. Récupération de données réelles via Hugging Face...")
    # Dataset célèbre de SMS Spam (UCI) disponible sur Hugging Face
    # C'est le standard utilisé en NLP pour le spam/phishing.
    dataset = load_dataset("ucirvine/sms_spam", split="train")
    
    # On le convertit en DataFrame Pandas
    df = pd.DataFrame(dataset)
    
    # Le dataset original a des labels 'ham' (legit) et 'spam' (phishing)
    df['label'] = df['label'].map({0: 'legit', 1: 'phish'})
    df = df.rename(columns={'sms': 'text'})
    
    print(f"   -> {len(df)} messages réels récupérés.")
    print(f"   -> Répartition : {df['label'].value_counts().to_dict()}")
    return df

def get_kaggle_data():
    print("2. Récupération de données via Kaggle (simulation)...")
    # Pour Kaggle, on utilise l'API kaggle (pip install kaggle)
    # Ici, on simule la lecture d'un CSV téléchargé manuellement
    print("   -> (Note: nécessite le fichier kaggle.json dans ~/.kaggle/)")
    # df = pd.read_csv("kaggle_phishing_dataset.csv")
    pass

if __name__ == "__main__":
    df_hf = get_huggingface_data()
    # df_kaggle = get_kaggle_data()
    
    # Sauvegarde pour l'utiliser dans le pipeline
    df_hf.to_csv("data/real_messages.csv", index=False)
    print("Données sauvegardées dans data/real_messages.csv")