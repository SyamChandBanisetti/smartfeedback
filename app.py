import io
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from itertools import zip_longest
from sklearn.feature_extraction.text import CountVectorizer
from agno.presets import SummaryAgent

# UI Setup
st.set_page_config(page_title="🧠 Feedback Analyzer", layout="wide")
st.title("🧠 Smart Feedback Analyzer")
st.caption("Built with Streamlit + ADK (Agno Development Kit)")

# Load AI Agent
agent = SummaryAgent()
agent_context = ""

# Helper: Chunk questions into groups of 3
def chunked(iterable, size=3):
    args = [iter(iterable)] * size
    return zip_longest(*args)

@st.cache_data
def extract_keywords(text_list, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(text_list)
    return list(vectorizer.get_feature_names_out())

@st.cache_data
def classify_sentiment(text):
    text = text.lower()
    if any(w in text for w in ["good", "great", "excellent", "love", "awesome"]):
        return "POSITIVE", "Positive sentiment"
    elif any(w in text for w in ["bad", "poor", "terrible", "hate", "worst"]):
        return "NEGATIVE", "Negative sentiment"
    elif text.strip() == "":
        return "NEUTRAL", "Empty"
    else:
        return "NEUTRAL", "No strong sentiment"

# Upload Section
uploaded_file = st.file_uploader("📂 Upload a Feedback CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    ignore_cols = ["timestamp", "email", "name", "id"]
    questions = [col for col in text_cols if col.lower() not in ignore_cols]

    st.markdown("## 🔍 Feedback Breakdown by Question")
    summary_data, response_data = [], []

    for group in chunked(questions):
        cols = st.columns(3)
        for i, question in enumerate(group):
            if not question:
                continue

            with cols[i]:
                st.subheader(f"❓ {question}")
                responses = df[question].dropna().astype(str).tolist()
                sentiments = [classify_sentiment(r) for r in responses]
                labels = [s[0] for s in sentiments]
                reasons = [s[1] for s in sentiments]

                count = Counter(labels)
                total = len(responses)
                pos = count.get("POSITIVE", 0)
                neg = count.get("NEGATIVE", 0)
                neu = count.get("NEUTRAL", 0)

                pie_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [pos, neg, neu]
                })

                # Charts
                st.plotly_chart(px.pie(pie_df, values="Count", names="Sentiment", hole=0.4), use_container_width=True)
                st.plotly_chart(px.bar(pie_df, x="Sentiment", y="Count", text="Count"), use_container_width=True)

                # Summary & Keywords
                keywords = extract_keywords(responses)
                st.markdown("🔑 **Top Keywords:** " + ", ".join(keywords))
                st.markdown(f"📝 **Total Responses:** {total}")
                st.markdown(f"👍 Positive: {round((pos/total)*100, 1)}%")
                st.markdown(f"👎 Negative: {round((neg/total)*100, 1)}%")
                st.markdown(f"😐 Neutral: {round((neu/total)*100, 1)}%")

                summary_data.append({
                    "Question": question,
                    "Total Responses": total,
                    "Positive %": round((pos / total) * 100, 1),
                    "Negative %": round((neg / total) * 100, 1),
                    "Neutral %": round((neu / total) * 100, 1),
                    "Keywords": ", ".join(keywords)
                })

                for r, l, rsn in zip(responses, labels, reasons):
                    response_data.append({
                        "Question": question,
                        "Response": r,
                        "Sentiment": l,
                        "Reason": rsn
                    })

                with st.expander("📋 Show Sample Responses"):
                    st.dataframe(pd.DataFrame({
                        "Response": responses,
                        "Sentiment": labels,
                        "Reason": reasons
                    }).head(10), use_container_width=True)

    # Update Agent Context
    agent_context = "\n".join(
        f"{row['Question']} -> Pos: {row['Positive %']}%, Neg: {row['Negative %']}%, Keywords: {row['Keywords']}"
        for row in summary_data
    )
    agent.update_context(agent_context)

    # Download Excel
    st.markdown("## 📥 Export Full Report")
    summary_df = pd.DataFrame(summary_data)
    response_df = pd.DataFrame(response_data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        response_df.to_excel(writer, sheet_name="Responses", index=False)
    output.seek(0)

    st.download_button(
        label="📁 Download Excel Report",
        data=output,
        file_name="feedback_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # AI Assistant Chat
    st.markdown("## 🤖 Chat with Feedback Assistant")
    user_input = st.text_input("💬 Ask something like: 'Which area needs improvement?'")

    if user_input:
        with st.spinner("🧠 Thinking..."):
            response = agent.query(user_input)
        st.success("Agent says:")
        st.markdown(response)

else:
    st.info("📤 Upload a feedback CSV file to get started.")
