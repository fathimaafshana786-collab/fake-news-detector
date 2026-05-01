 
# Fake News Detector

A hybrid AI system that detects fake news by combining BERT-based classification with LLM-powered reasoning. The goal was to build something that doesn't just label an article as fake or real, but actually explains the reasoning behind that decision.

---

## Why This Project

Misinformation is everywhere and most detection tools treat it as a binary problem — fake or not fake. That's rarely how it works in practice. Some articles are misleading without being outright false. Some are sensationalized but factually accurate. This project tries to handle that nuance by combining two models that approach the problem differently.

---

## How It Works

The system runs two models in sequence:

**BERT** reads the article and classifies it as fake or real based on patterns it learned during fine-tuning. It's fast and confident but limited to what it was trained on.

**Gemma3** takes the article along with BERT's verdict and reasons through it independently. It looks for things like unsupported claims, sensationalized language, missing sources, and logical inconsistencies. Then it explains its conclusion in plain language.

The final verdict is based on whether the two models agree:

- Both say fake — high confidence fake
- Both say real — high confidence real
- They disagree — the system flags it as uncertain rather than forcing a conclusion

That last case matters. Saying "uncertain" is more honest than picking a side when the evidence is not clear.

---

## Results

Trained on 10,000 balanced news articles across fake and real categories.

```
Train Accuracy : 99.90%
Test Accuracy  : 99.85%
F1 Score       : 99.85%

              Precision    Recall    F1
Real News        1.00        1.00    1.00
Fake News        1.00        1.00    1.00
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| BERT (bert-base-uncased) | News classification |
| Ollama + Gemma3:4b | Local LLM reasoning |
| PyTorch | Model training |
| HuggingFace Transformers | BERT implementation |
| Streamlit | Web interface |
| Scikit-learn | Evaluation metrics |

---

## Getting Started

**Prerequisites**

- Python 3.10
- Ollama installed and running — https://ollama.com
- Gemma3 model: `ollama pull gemma3:4b`

**Installation**

```bash
git clone https://github.com/fathimaafshana786-collab/fake-news-detector.git
cd fake-news-detector

py -3.10 -m venv news_env
news_env\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
fake-news-detector/
├── data/               cleaned dataset files
├── bert_model/         saved fine-tuned BERT weights
├── train_bert.py       training pipeline
├── ollama_explain.py   Gemma3 reasoning module
├── backend.py          core logic connecting both models
├── app.py              Streamlit web interface
└── requirements.txt    dependencies
```

---

## Known Limitations

This project was built to learn and demonstrate a hybrid AI approach, not to replace professional fact-checking tools. A few honest limitations worth knowing:

- The model performs best on political news, which is what the training data focused on. It may struggle with science, sports, or business articles.
- Both models have knowledge cutoffs. Very recent events may not be handled correctly.
- Short or out-of-context statements are not what this system was designed for. It works best on full articles or substantial paragraphs.

---

## What I Would Do Differently

- Use a more diverse training dataset covering multiple news categories
- Add a confidence threshold below which the system refuses to give a verdict
- Fine-tune Gemma3 on fact-checking specific data rather than using it zero-shot
- Deploy on HuggingFace Spaces for public access

---

## License

MIT License. Free to use, modify, and build on.