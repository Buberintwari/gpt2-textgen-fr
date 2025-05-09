---
title: Générateur de texte GPT-2 en français
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---
## 🎥 Démonstration Vidéo

Cliquez sur l'image ci-dessous pour regarder la vidéo de démonstration sur YouTube :

[![Regarder la vidéo](https://img.youtube.com/vi/XSRaZlfsaI8/0.jpg)](https://youtu.be/XSRaZlfsaI8?si=Ew5cAPxf-c-kd15Y)

## 🚀 Tester le modèle en ligne

👉 [Accéder à l'application sur Hugging Face Spaces](https://huggingface.co/spaces/Buberintwari/gpt2-fr-generateur)


# Générateur de texte en français avec GPT-2

Ce projet utilise le modèle `asi/gpt-fr-cased-small` pour générer du texte en français à partir d'une phrase ou d'un début de paragraphe.

Modèle utilisé : GPT-2 fine-tuné en français  
Interface simple via Gradio  
Déployé sur Hugging Face Spaces  
Type d'apprentissage : Self-Supervised Learning (SSL)

## Comment ça fonctionne ?
L'utilisateur entre un texte. Le modèle prédit la suite du texte en se basant sur ce qu'il a appris à partir de corpus francophones.

## Exemple
> **Entrée** : Un jour, un petit robot décida de...  
> **Sortie** : ... partir à l’aventure pour découvrir le monde, malgré les risques et les mystères qui l'attendaient.

## Objectif de projet
Comprendre et démontrer un cas concret de Self-Supervised Learning appliqué au traitement automatique du langage naturel (NLP) avec GPT-2.

---

Ce projet est développé et testé dans **Google Colab** puis déployé via **Gradio** sur **Hugging Face Spaces**.
