import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Chargement du modèle GPT-2 français
#model_name = "dbddv01/gpt2-french-small"
model_name = "asi/gpt-fr-cased-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fonction de génération
def generate_text(prompt, max_length):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=40,
        top_p=0.92,
        temperature=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interface Gradio
description = """
## ✍️ Conseils pour un bon prompt
Pour de meilleurs résultats, utilisez des phrases complètes ou des instructions claires.  
**Exemples :**  
- "Raconte une petite histoire sur un oiseau qui apprend à voler."  
- "Complète ce texte : L'oiseau bleu survolait les montagnes..."  
"""

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, placeholder="Ex : Raconte une histoire sur un oiseau...", label="Prompt"),
        gr.Slider(50, 300, value=150, step=10, label="Longueur maximale du texte")
    ],
    outputs=gr.Textbox(label="Texte généré"),
    title="📝 Générateur de texte GPT-2 (français)",
    description=description
)

if __name__ == "__main__":
    demo.launch()
