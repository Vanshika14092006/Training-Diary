# StudyMate AI 📚

StudyMate AI is an AI-powered study assistant that transforms notes into:
- Student-friendly summaries
- Important exam questions
- Multiple-choice questions (MCQs)
- A practical 3-day revision plan

## Tech stack
- Python
- Streamlit
- OpenAI API
- PyPDF

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## API key

Do not hard-code your API key.

For local testing, you can enter the key in the sidebar.

For Streamlit Community Cloud, add the secret:

```toml
OPENAI_API_KEY = "your-key-here"
```

Then deploy the repository and use the deployed Streamlit URL as the project's live demo.

## Hackathon demo flow

1. Open StudyMate AI.
2. Paste sample lecture notes or upload a TXT/PDF.
3. Click "Generate Study Pack".
4. Show Summary.
5. Show Key Questions.
6. Show MCQs.
7. Show Revision Plan.


