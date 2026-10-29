import streamlit as st
import pandas as pd
import joblib
import re
import os

st.set_page_config(
    page_title="Support Ticket Classification",
    page_icon="🎫",
    layout="wide"
)

MODEL_PATH = "models/linear_svm_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

svm_model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def assign_priority(ticket_text):
    text = ticket_text.lower()

    high_priority_keywords = [
        "critical", "urgent", "emergency", "system down",
        "server down", "complete outage", "security breach",
        "data breach", "unable to work", "production down"
    ]

    medium_priority_keywords = [
        "not working", "error", "failed", "failure",
        "unable to access", "access denied", "issue",
        "problem", "intermittent", "slow"
    ]

    if any(keyword in text for keyword in high_priority_keywords):
        return "High"
    elif any(keyword in text for keyword in medium_priority_keywords):
        return "Medium"
    else:
        return "Low"

def classify_ticket(ticket_text):
    cleaned_ticket = clean_text(ticket_text)
    ticket_vector = tfidf.transform([cleaned_ticket])
    predicted_category = svm_model.predict(ticket_vector)[0]
    predicted_priority = assign_priority(ticket_text)

    return predicted_category, predicted_priority

st.title("🎫 AI-Powered Support Ticket Classification & Prioritization")

st.write(
    "Automatically classify IT support tickets and assign priority "
    "using Machine Learning and rule-based classification."
)

st.divider()

st.header("Single Ticket Classification")

ticket_text = st.text_area(
    "Enter your support ticket:",
    placeholder="Example: Unable to access the company server. The application shows an error.",
    height=150
)

if st.button("Classify Ticket", type="primary"):
    if ticket_text.strip():
        category, priority = classify_ticket(ticket_text)

        st.subheader("Classification Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Category", category)

        with col2:
            st.metric("Assigned Priority", priority)

    else:
        st.warning("Please enter a support ticket before classification.")
        
st.divider()

st.divider()

st.header("Batch Ticket Classification")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing support tickets",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_file.seek(0)
        batch_df = pd.read_csv(uploaded_file)

        if batch_df.empty:
            st.error("The uploaded CSV file is empty.")

        elif "Document" not in batch_df.columns:
            st.error("CSV file must contain a 'Document' column.")

        else:
            batch_df["Document"] = (
                batch_df["Document"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            valid_df = batch_df[batch_df["Document"] != ""].copy()

            removed_tickets = len(batch_df) - len(valid_df)

            if valid_df.empty:
                st.error("No valid support tickets found in the uploaded file.")

            else:
                st.subheader("Uploaded Dataset")
                st.dataframe(valid_df.head())

                if removed_tickets > 0:
                    st.warning(
                        f"{removed_tickets} empty ticket records were removed."
                    )

                if st.button("Classify All Tickets"):

                    predictions = valid_df["Document"].apply(classify_ticket)

                    valid_df["Predicted Category"] = predictions.apply(
                        lambda x: x[0]
                    )

                    valid_df["Assigned Priority"] = predictions.apply(
                        lambda x: x[1]
                    )

                    st.subheader("Classification Results")
                    st.dataframe(valid_df)

                    csv_data = valid_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Classified Tickets",
                        data=csv_data,
                        file_name="classified_support_tickets.csv",
                        mime="text/csv"
                    )

                    st.divider()

                    st.header("Ticket Analytics Dashboard")

                    total_tickets = len(valid_df)

                    high_tickets = (
                        valid_df["Assigned Priority"] == "High"
                    ).sum()

                    medium_tickets = (
                        valid_df["Assigned Priority"] == "Medium"
                    ).sum()

                    low_tickets = (
                        valid_df["Assigned Priority"] == "Low"
                    ).sum()

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Total Tickets", total_tickets)
                    col2.metric("High Priority", high_tickets)
                    col3.metric("Medium Priority", medium_tickets)
                    col4.metric("Low Priority", low_tickets)

                    st.subheader("Ticket Category Distribution")

                    category_counts = (
                        valid_df["Predicted Category"].value_counts()
                    )

                    st.bar_chart(
                        category_counts,
                        horizontal=True
                    )

                    st.subheader("Ticket Priority Distribution")

                    priority_counts = (
                        valid_df["Assigned Priority"]
                        .value_counts()
                        .reindex(["High", "Medium", "Low"], fill_value=0)
                    )

                    st.bar_chart(
                        priority_counts,
                        horizontal=True
                    )

    except Exception as e:
        st.error(f"Unable to process the uploaded CSV file: {e}")