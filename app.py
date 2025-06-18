Guide: Building and Deploying Your Streamlit Feedback Analyzer App
This guide will walk you through setting up your project locally and then deploying it online using GitHub and Streamlit Community Cloud.

Step 1: Set Up Your Project Files
You'll need two main files in your project directory (which will become your GitHub repository).

1. app.py (Your Streamlit Application Code)
This is the core of your application. It includes all the Streamlit UI elements, the logic for reading your data, performing sentiment analysis and summarization with the Gemini API, and displaying the results.

I've made sure this code is concise, efficient, and ready for deployment.

import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import re # Used for robust parsing of LLM output

# --- Setup ---
# Your Gemini API key must be securely added to Streamlit Cloud secrets.
# On Streamlit Cloud, go to your app settings (three dots menu -> "Manage app" -> "Secrets")
# Add a new secret with Key: GOOGLE_API_KEY and Value: "YOUR_GEMINI_API_KEY_HERE"
try:
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add 'GOOGLE_API_KEY' to your app's secrets for deployment.")
    st.info("For local testing, you can create a `.streamlit/secrets.toml` file with `GOOGLE_API_KEY='YOUR_KEY'`")
    st.stop() # Stop the app execution if the key is missing

genai.configure(api_key=GEMINI_API_KEY)
# Initialize the Gemini 2.0 Flash model
model = genai.GenerativeModel("gemini-2.0-flash")

# Set basic Streamlit page configuration
st.set_page_config(
    page_title="Gemini Feedback Analyzer",
    layout="wide", # Use wide layout for better visualization
    page_icon="✨"
)

# --- Helper Functions (using @st.cache_data for performance) ---

@st.cache_data(show_spinner=False)
def get_sentiment_and_score(text_series_input):
    """
    Analyzes the sentiment of each text in a Pandas Series and assigns a score.
    Score: -1 (Negative), 0 (Neutral), 1 (Positive)
    This function processes each text individually through the LLM.
    """
    sentiments, scores = [], []
    # Only process non-empty string responses
    texts_to_process = text_series_input.dropna().astype(str)

    # Use a progress bar for user feedback during LLM calls
    progress_bar = st.progress(0, text="Analyzing sentiments...")
    total_texts = len(texts_to_process)

    for i, text in enumerate(texts_to_process):
        if not text.strip(): # Skip if text is empty after stripping
            sentiments.append("Neutral")
            scores.append(0.0)
            continue

        prompt = f"""Analyze the sentiment of the following feedback and classify it as Positive, Negative, or Neutral.
        Then, provide a sentiment score: -1 for Negative, 0 for Neutral, and 1 for Positive.
        Return the output in the exact format: Sentiment: [Sentiment], Score: [Score]
        Feedback: "{text}"\n"""
        try:
            response = model.generate_content(prompt)
            result = response.text.strip()

            sentiment = "Unknown"
            score = np.nan

            # Robust parsing using regex
            sentiment_match = re.search(r"Sentiment:\s*([A-Za-z]+)", result)
            score_match = re.search(r"Score:\s*([-+]?\d*\.?\d+)", result)

            if sentiment_match:
                sentiment = sentiment_match.group(1).strip()
            if score_match:
                try:
                    score = float(score_match.group(1).strip())
                except ValueError:
                    pass # Keep score as NaN if conversion fails

            sentiments.append(sentiment)
            scores.append(score)
        except Exception as e:
            # Handle API errors gracefully
            st.warning(f"Error processing text (first 50 chars: '{text[:50]}...'): {e}")
            sentiments.append("Error")
            scores.append(np.nan)

        progress_bar.progress((i + 1) / total_texts, text=f"Analyzing sentiments... ({i+1}/{total_texts})")
    progress_bar.empty() # Clear progress bar after completion

    # Return results as Pandas Series, maintaining original index alignment for merging
    return pd.Series(sentiments, index=texts_to_process.index), pd.Series(scores, index=texts_to_process.index)


@st.cache_data(show_spinner=False)
def summarize_feedback_themes(feedback_text_list, creativity=0.4):
    """
    Summarizes a list of feedback texts into key themes using the LLM.
    """
    joined_feedback = "\n".join(feedback_text_list).strip()
    if not joined_feedback:
        return "No sufficient feedback to summarize themes."

    prompt = f"""Review the following customer feedback entries. Identify and list the top 3-5 most important recurring themes or topics discussed.
    Present them as a bullet-point list.
    Feedback:
    {joined_feedback}

    Themes:"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating summary themes: {e}")
        return "Could not generate key themes."

@st.cache_data(show_spinner=False)
def generate_recommendations(negative_feedback_summary, creativity=0.6):
    """
    Generates actionable recommendations based on a summary of negative feedback.
    """
    if not negative_feedback_summary.strip() or "No sufficient feedback" in negative_feedback_summary:
        return "No specific negative feedback identified to generate recommendations."

    prompt = f"""Based on the following summary of negative feedback, provide 3-5 clear, actionable recommendations for improvement.
    Summary of negative feedback:
    {negative_feedback_summary}

    Actionable Recommendations:"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=creativity)
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return "Could not generate recommendations."

def plot_wordcloud(text_data):
    """Generates and displays a word cloud from given text data."""
    if not text_data or not isinstance(text_data, list):
        st.info("No text data available for word cloud.")
        return

    text = " ".join([str(item) for item in text_data if pd.notna(item) and str(item).strip() != ""])
    if not text.strip():
        st.info("Not enough valid text to generate a word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


# --- Main Application Logic ---

st.title("✨ AI-Powered Feedback Analyzer")
st.markdown("Upload your customer feedback data (CSV or Excel) to gain instant insights into sentiment, key themes, and actionable recommendations using **Gemini 2.0 Flash**.")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload your feedback file", type=["csv", "xlsx"])

if uploaded_file:
    # Load data based on file type
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}. Please ensure it's a valid CSV or Excel format.")
        st.stop()

    st.subheader("📄 Raw Data Preview (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

    # Identify potential text columns for analysis
    # A column is considered 'text' if it's an object/string type and average length > 10 characters
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].astype(str).apply(len).mean() > 10]

    if not text_columns:
        st.warning("No suitable open-ended text columns found for analysis. Please ensure your file contains columns with written feedback.")
        st.stop()

    st.markdown("---")
    st.subheader("⚙️ Analysis Configuration")
    selected_column = st.selectbox(
        "Select the text column containing feedback for AI analysis:",
        text_columns,
        help="Choose the column that holds customer comments, reviews, or open-ended responses."
    )

    max_rows_for_llm = st.slider(
        "Limit rows for AI processing (for speed)",
        min_value=10, max_value=500, value=100, step=10,
        help="The AI will process a sample of this many rows from the selected column. Higher numbers provide more detailed analysis but take longer."
    )

    llm_creativity = st.slider(
        "AI Creativity (Temperature for Summaries/Recommendations)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.1,
        help="Controls the randomness of AI-generated summaries and recommendations. Lower values (e.g., 0.2-0.5) are more focused and factual. Higher values (e.g., 0.7-1.0) are more diverse but can be less precise."
    )

    st.markdown("---")
    if st.button("🚀 Run AI Analysis", help="Click to start processing feedback with Gemini 2.0 Flash."):
        if selected_column:
            st.markdown("### 📈 Analysis Results")

            # Sample the responses for LLM processing
            responses_to_analyze = df[selected_column].dropna().sample(n=min(len(df[selected_column].dropna()), max_rows_for_llm), random_state=42)
            if responses_to_analyze.empty:
                st.info("No valid responses found in the selected sample for analysis.")
                st.stop()

            # --- Sentiment Analysis ---
            st.subheader("1. Sentiment Breakdown")
            # Apply sentiment analysis to the sampled responses
            with st.spinner(f"Running sentiment analysis on {len(responses_to_analyze)} responses..."):
                sentiments, scores = get_sentiment_and_score(responses_to_analyze)

            # Create a temporary DataFrame for sampled results for display
            sampled_analysis_df = pd.DataFrame({
                'text': responses_to_analyze,
                'sentiment': sentiments,
                'score': scores
            })

            # Display sentiment distribution
            if not sampled_analysis_df.empty:
                sentiment_counts = sampled_analysis_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                fig_sentiment = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                                       title="Overall Sentiment Distribution",
                                       color_discrete_map={"Positive":"#28a745", "Negative":"#dc3545", "Neutral":"#ffc107", "Unknown":"#6c757d", "Error":"#6c757d"})
                st.plotly_chart(fig_sentiment, use_container_width=True)

                avg_score = sampled_analysis_df['score'].mean()
                if not np.isnan(avg_score):
                    st.info(f"**Average Sentiment Score:** {avg_score:.2f} (from -1 to 1; higher is more positive)")
            else:
                st.info("No sentiment data generated for visualization.")

            # --- Word Cloud ---
            st.subheader("2. Key Themes Word Cloud")
            with st.spinner("Generating word cloud..."):
                plot_wordcloud(responses_to_analyze.tolist())


            # --- Summary of Themes ---
            st.subheader("3. AI-Generated Key Themes Summary")
            with st.spinner("Summarizing key themes..."):
                summary_themes = summarize_feedback_themes(responses_to_analyze.tolist(), creativity=llm_creativity)
                st.success(summary_themes)

            # --- Actionable Recommendations from Negative Feedback ---
            st.subheader("4. Actionable Recommendations (from Negative Feedback)")
            negative_responses_sampled = sampled_analysis_df[sampled_analysis_df['sentiment'] == 'Negative']['text'].dropna().tolist()

            if negative_responses_sampled:
                with st.spinner(f"Generating recommendations from {len(negative_responses_sampled)} negative responses..."):
                    negative_summary_for_recs = summarize_feedback_themes(negative_responses_sampled, creativity=llm_creativity)
                    recommendations = generate_recommendations(negative_summary_for_recs, creativity=llm_creativity)
                    st.warning(recommendations)
            else:
                st.info("No negative feedback identified in the sample to generate recommendations.")

else:
    # Initial message when no file is uploaded
    st.info("Upload a CSV or Excel file to begin analyzing your feedback!")
    st.markdown("---")
    st.markdown("### Example Data Structure:")
    # Display a small sample dataframe to show expected input format
    sample_data = pd.DataFrame({
        "Timestamp": ["2024-06-01", "2024-06-02", "2024-06-03"],
        "Customer Feedback": [
            "The new app update is fantastic! So much faster and intuitive.",
            "Customer support was very slow, took ages to get a reply. Disappointed.",
            "Product features are good, but the pricing is too high."
        ],
        "Rating (1-5)": [5, 2, 4]
    })
    st.dataframe(sample_data, use_container_width=True)
    st.markdown("*(Your file can have many columns, but at least one should contain open-ended text feedback)*")

