#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
csv_file_path = "chat_history_visualization.csv"
df = pd.read_csv(csv_file_path)


# Function to generate a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)  # Display the word cloud in Streamlit

def visualize_diversity(df, user_col, bot_col, timestamp_col):
    # Combine user and bot responses into a single column
    responses = pd.concat([df[user_col], df[bot_col]])

    # Convert timestamp to datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Group by timestamp and count unique responses
    diversity_data = responses.groupby(df[timestamp_col]).nunique()

    # Plotting
    st.header("Diversity of Responses Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(diversity_data.index, diversity_data, marker='o')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Number of Unique Responses")
    ax.set_title("Diversity of Responses Over Time")
    st.pyplot(fig)
    
def analyze_errors(df):
    # Create a new column for end_continue_result performance
    df['End_Continue_Performance'] = (df['end_continue_result'] == df['GroundTruthCC']).astype(int)

    # Create a new column for wiki_chat_result performance
    df['Wiki_Performance'] = (df['wiki_chat_result'] == df['GroundtruthWC']).astype(int)

    # Plotting for End_Continue_Performance
    st.subheader("End_Continue_Performance Distribution")
    st.bar_chart(df['End_Continue_Performance'].value_counts())

    # Plotting for Wiki_Performance
    st.subheader("Wiki_Performance Distribution")
    st.bar_chart(df['Wiki_Performance'].value_counts())

def query_response_relevance(df):
    mismatched_conversations_cc = df[df['GroundTruthCC'] != df['end_continue_result']]

    # Create a table plot
    st.subheader("Mismatched Conversations - GroundTruthCC vs End_Continue_Result")
    st.table(mismatched_conversations_cc[['user_input', 'bot_response', 'end_continue_result', 'GroundTruthCC']])


# Example usage:
# Assuming your DataFrame is named 'df'
st.header("Error Analysis")
analyze_errors(df)

st.header("Queries and Resonponse Analysis")
query_response_relevance(df)

# Example: Display a Word Cloud for user responses
user_responses_text = ' '.join(df['user_input'])
st.header("Word Cloud for User Responses")
generate_word_cloud(user_responses_text)

# Example: Display a chart
st.header("Conversation Topics Analysis")
visualize_diversity(df, user_col="user_input", bot_col="bot_response", timestamp_col="timestamp")


# In[ ]:




