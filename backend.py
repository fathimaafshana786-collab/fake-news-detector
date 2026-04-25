 # ============================================
# PHASE 5 - Backend (BERT + Ollama Combined)
# ============================================

# Step 1: Import libraries
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import ollama
import re

# Step 2: Load BERT model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert_model")
model     = BertForSequenceClassification.from_pretrained("bert_model")
model.eval()
print("✅ BERT model loaded!")

# Step 3: BERT prediction function
def bert_predict(text):
    """
    Takes article text
    Returns verdict (0=Real, 1=Fake) and confidence
    """
    # Tokenize
    inputs = tokenizer(
        text,
        max_length     = 128,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        verdict = torch.argmax(probs, dim=1).item()
        confidence = probs[0][verdict].item()

    return verdict, confidence

# Step 4: Ollama explanation function
def ollama_explain(text, bert_verdict, bert_confidence):
    """
    Takes article + BERT result
    Returns explanation from Gemma3
    """
    prompt = f"""
You are a professional fact-checking assistant.

A BERT AI model has analyzed this news article and determined:
- Verdict    : {"FAKE NEWS" if bert_verdict == 1 else "REAL NEWS"}
- Confidence : {bert_confidence:.1%}

Now YOU analyze this article and explain why it is fake or real.

Article:
{text[:1000]}

Respond in this exact format:
VERDICT: [FAKE or REAL]
CONFIDENCE: [HIGH or MEDIUM or LOW]
REASONS:
- [Reason 1]
- [Reason 2]
- [Reason 3]
SUSPICIOUS PHRASES:
- [Phrase 1]
- [Phrase 2]
"""

    response = ollama.chat(
        model    = "gemma3:4b",
        messages = [
            {
                "role"    : "user",
                "content" : prompt
            }
        ]
    )

    return response['message']['content']

# Step 5: Agreement logic function
def get_final_verdict(bert_verdict, bert_confidence, llm_response):
    """
    Combines BERT + Gemma3 results
    Returns final unified verdict
    """
    # Check what Gemma3 said
    llm_response_upper = llm_response.upper()

    if "VERDICT: FAKE" in llm_response_upper:
        llm_verdict = 1  # Fake
    elif "VERDICT: REAL" in llm_response_upper:
        llm_verdict = 0  # Real
    else:
        llm_verdict = -1  # Uncertain

    # Agreement logic
    if bert_verdict == llm_verdict:
        # Both agree!
        if bert_verdict == 1:
            final = "FAKE NEWS ❌"
        else:
            final = "REAL NEWS ✅"
        agreement = "HIGH (Both BERT and AI agree!)"
    else:
        # They disagree!
        final     = "UNCERTAIN ⚠️"
        agreement = "LOW (BERT and AI disagree - needs review)"

    return final, agreement

# Step 6: Main analyze function
def analyze_article(text):
    """
    Full pipeline:
    Article → BERT → Gemma3 → Final verdict
    """
    print("\n" + "=" * 50)
    print("Analyzing article...")
    print("=" * 50)

    # BERT prediction
    print("\n🤖 Running BERT analysis...")
    bert_verdict, bert_confidence = bert_predict(text)
    print(f"BERT Verdict     : {'FAKE' if bert_verdict == 1 else 'REAL'}")
    print(f"BERT Confidence  : {bert_confidence:.1%}")

    # Gemma3 explanation
    print("\n🦙 Running Gemma3 analysis...")
    llm_response = ollama_explain(text, bert_verdict, bert_confidence)

    # Final verdict
    final_verdict, agreement = get_final_verdict(
        bert_verdict, bert_confidence, llm_response
    )

    # Return everything
    return {
        "bert_verdict"    : "FAKE" if bert_verdict == 1 else "REAL",
        "bert_confidence" : f"{bert_confidence:.1%}",
        "llm_explanation" : llm_response,
        "final_verdict"   : final_verdict,
        "agreement"       : agreement
    }

# Step 7: Test the full pipeline
if __name__ == "__main__":
    test_article = """
    Scientists have discovered that drinking 
    coffee every morning can cure cancer completely.
    Doctors are hiding this secret from the public
    to keep selling expensive medications.
    """

    result = analyze_article(test_article)

    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    print(f"BERT Verdict     : {result['bert_verdict']}")
    print(f"BERT Confidence  : {result['bert_confidence']}")
    print(f"Final Verdict    : {result['final_verdict']}")
    print(f"Agreement        : {result['agreement']}")
    print(f"\nAI Explanation:\n{result['llm_explanation']}")
    print("=" * 50)
    print("✅ Backend working perfectly!")
 
 
