# evaluate_and_plot.py
# Script to evaluate a PII NER model and plot a confusion matrix
# Groups similar PII types into broader categories for readability

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_from_disk

# 1. Setup Paths
MODEL_DIR = os.path.join("models", "pii_ner")
DATA_DIR = os.path.join("data", "processed", "ai4privacy_tokenized")
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Define Category Mapping (Grouping similar PII)
CATEGORY_MAP = {
    # Names
    "FIRSTNAME": "NAME", "LASTNAME": "NAME", "MIDDLENAME": "NAME", "USERNAME": "NAME", "ACCOUNTNAME": "NAME",
    
    # Financial / Crypto
    "BITCOINADDRESS": "FINANCE", "LITECOINADDRESS": "FINANCE", "ETHEREUMADDRESS": "FINANCE",
    "IBAN": "FINANCE", "CREDITCARDNUMBER": "FINANCE", "CVV": "FINANCE", "AMOUNT": "FINANCE",
    "BIC": "FINANCE", "CURRENCY": "FINANCE", "CURRENCYCODE": "FINANCE", "CURRENCYSYMBOL": "FINANCE",
    
    # IDs / Tech
    "SSN": "ID", "PASSPORTNUMBER": "ID", "DRIVERLICENSE": "ID", "TAXID": "ID",
    "IP": "TECH", "IPV4": "TECH", "IPV6": "TECH", "MAC": "TECH", "USERAGENT": "TECH",
    
    # Location
    "STREET": "ADDRESS", "CITY": "ADDRESS", "COUNTY": "ADDRESS", "STATE": "ADDRESS", 
    "ZIPCODE": "ADDRESS", "BUILDINGNUMBER": "ADDRESS", "NEARBYGPSCOORDINATE": "ADDRESS",
    
    # Contact
    "EMAIL": "CONTACT", "PHONENUMBER": "CONTACT", "PHONEIMEI": "CONTACT",
    
    # Other
    "O": "O"
}

def clean_and_map_tag(tag):
    """
    1. Removes B- or I- prefix.
    2. Maps specific entity to broad category (e.g. FIRSTNAME -> NAME).
    """
    if tag == "O":
        return "O"
    
    # Remove B- / I-
    core_label = tag.split("-")[1]
    
    # Return mapped category, or 'OTHER' if not in map
    return CATEGORY_MAP.get(core_label, "OTHER")

def generate_grouped_confusion_matrix():
    print("🔹 Loading model for grouped evaluation...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    dataset = load_from_disk(DATA_DIR)
    eval_dataset = dataset["eval"]
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    
    print("🔹 Running predictions...")
    predictions, labels, _ = trainer.predict(eval_dataset)
    predictions = np.argmax(predictions, axis=2)
    
    id2label = model.config.id2label

    true_labels = []
    pred_labels = []

    print("🔹 Grouping labels...")
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            label_id = labels[i][j]
            pred_id = predictions[i][j]
            
            if label_id != -100:
                true_cat = clean_and_map_tag(id2label[label_id])
                pred_cat = clean_and_map_tag(id2label[pred_id])
                
                true_labels.append(true_cat)
                pred_labels.append(pred_cat)

    # Sort labels to keep matrix consistent
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    # Ensure 'O' is last for better visualization
    if 'O' in unique_labels:
        unique_labels.remove('O')
        unique_labels.append('O')
        
    print(f"🔹 Plotting Matrix with categories: {unique_labels}")
    
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', 
                xticklabels=unique_labels, yticklabels=unique_labels, 
                cmap='Blues', annot_kws={"size": 10})
    
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.title('Confusion Matrix (Grouped by Category)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "confusion_matrix_grouped.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved readable matrix to {out_path}")

if __name__ == "__main__":
    generate_grouped_confusion_matrix()