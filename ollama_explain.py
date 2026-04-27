 # ============================================
# PHASE 4 - Ollama LLaMA 3 Explanation
# ============================================

import ollama

def explain_news(article, bert_verdict, bert_confidence):
    """
    Takes article + BERT result
    Returns explanation from LLaMA 3
    """

    prompt = f"""
You are a professional fact-checking assistant.
Today's date is April 27, 2026.
You have knowledge of events up to early 2025.

A BERT AI model has analyzed this news article:
- Verdict    : {"FAKE NEWS" if bert_verdict == 1 else "REAL NEWS"}
- Confidence : {bert_confidence:.1%}

Important rules:
- Do NOT flag something as fake just because
  it mentions dates in 2025 or 2026
- Do NOT flag simple factual statements as fake
- Focus on logical inconsistencies and
  misinformation patterns only
- If BERT says REAL with high confidence
  lean towards REAL unless clearly wrong

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

    # Call Ollama
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


# Test it
if __name__ == "__main__":
    test_article = """
    Trump wins 2024 election by a landslide!
    Sources say he won every single state in 
    an unprecedented victory never seen before 
    in American history.
    """

    print("Testing Ollama connection...")
    print("=" * 50)

    result = explain_news(
        article         = test_article,
        bert_verdict    = 1,
        bert_confidence = 0.95
    )

    print(result)
    print("=" * 50)
    print("✅ Ollama is working perfectly!")
