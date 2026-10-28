"""
test_real_data.py
Test d'entraînement sur des données réelles (Hugging Face) avec Balancing.
Ce script prouve la faisabilité de l'extension sans impacter l'application Streamlit.
Exécution : python test_real_data.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.pipeline import make_pipeline

def main():
    print("1. Chargement des données réelles...")
    try:
        df = pd.read_csv("data/real_messages.csv")
    except FileNotFoundError:
        print("Erreur : data/real_messages.csv introuvable. Lancez d'abord data_collection.py")
        return

    print("Colonnes disponibles :", df.columns.tolist())
    
    # --- Correction du chargement des labels ---
    # On force la conversion en texte pour éviter les bugs de type (Arrow/Pandas)
    if 'label' in df.columns:
        labels_text = df['label'].astype(str).str.lower()
        # On considère comme phishing tout ce qui contient 'phish', 'spam' ou '1'
        df['y'] = labels_text.isin(['phish', 'spam', '1']).astype(int)
    elif 'target' in df.columns:
        df['y'] = df['target'].astype(int)
    else:
        raise ValueError("Colonne de label introuvable dans le CSV.")
    
    print(f"   -> Taille du dataset : {len(df)} messages.")
    print(f"   -> Répartition AVANT balancing : {df['y'].value_counts().to_dict()}")
    
    # 2. Split stratifié 
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['y'], test_size=0.25, stratify=df['y'], random_state=42
    )
    
    # 3. APPROCHE 1 : Balancing via class_weight (intégré au modèle)
    print("\n2. Entraînement avec class_weight='balanced' (intégré)...")
    pipe_lr = make_pipeline("lr").fit(X_train, y_train)
    preds_lr = pipe_lr.predict(X_test)
    print("\n--- Résultats Logistic Regression (class_weight='balanced') ---")
    print(classification_report(y_test, preds_lr, target_names=['Legit (Ham)', 'Phish (Spam)']))
    
    # 4. APPROCHE 2 : Data Augmentation via Oversampling (imblearn)
    print("\n3. Entraînement avec Oversampling (RandomOverSampler)...")
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import RandomOverSampler
        
        # On crée un pipeline spécifique avec l'oversampler
        base_pipe = make_pipeline("lr")
        pipe_oversample = ImbPipeline([
            ('tfidf', base_pipe.named_steps['tfidf']),
            ('oversample', RandomOverSampler(random_state=42)),
            ('clf', base_pipe.named_steps['clf'])
        ])
        
        pipe_oversample.fit(X_train, y_train)
        preds_os = pipe_oversample.predict(X_test)
        print("\n--- Résultats Logistic Regression (Oversampling) ---")
        print(classification_report(y_test, preds_os, target_names=['Legit (Ham)', 'Phish (Spam)']))
        
    except ImportError:
        print("   -> (Optionnel) Installe imbalanced-learn pour tester l'oversampling : pip install imbalanced-learn")

if __name__ == "__main__":
    main()