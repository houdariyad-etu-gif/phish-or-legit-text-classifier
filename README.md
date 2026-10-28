# Lab 04 — Phish or Legit?

Classification binaire de messages SMS-like (phishing-style vs legit) sur un dataset SYNTHÉTIQUE
(2 000 messages, 20 % phishing, marques fictives).

## Lancer
pip install -r requirements.txt
streamlit run app.py

## Évaluation
python -m src.evaluate   # -> results/metrics.json, failures.txt, phish_results.png

## App publique
https://phish-or-legit-classifier.streamlit.app

## Avertissement
Données synthétiques. L'outil aide une décision humaine, il ne bloque pas les messages.
