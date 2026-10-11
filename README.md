# 🤖 My Digital Twin

A Streamlit app that models my daily activity patterns (frequencies and
Markov-style transition probabilities) and lets me chat with a Gemini-powered
assistant that reasons over that data plus a small knowledge base.

## Run locally
1. `pip install -r requirements.txt`
2. Create a `.env` file: `GEMINI_API_KEY=your_key`
3. `streamlit run app.py`

## Test
`python -m pytest`

## Deploy (Streamlit Community Cloud)
Push to GitHub, create the app at share.streamlit.io with `app.py` as the
main file, and add `GEMINI_API_KEY` under Advanced settings → Secrets.