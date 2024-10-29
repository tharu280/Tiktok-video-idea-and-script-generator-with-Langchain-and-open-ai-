import os
from apikey import api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import io

os.environ['OPENAI_API_KEY'] = api_key


st.title('🎥 TikTok Video Idea and Script Generator')
prompt = st.text_input('Describe your video/video field (e.g., "toy cars", "travel tips", "cooking")')


idea_template = PromptTemplate(
    input_variables=['topic'],
    template='As a creative TikTok content creator, suggest a unique and engaging video idea about {topic}. '
             'Include a catchy title and a brief outline of the video content.'
)

script_template = PromptTemplate(
    input_variables=['title'],
    template='Write a script for the TikTok video titled "{title}". The script should be engaging and concise, '
             'suitable for a TikTok audience.'
)


idea_memory = ConversationBufferMemory(input_key='topic', memory_key='idea_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='script_history')


llm = OpenAI(temperature=0.9)
idea_chain = LLMChain(llm=llm, prompt=idea_template, verbose=True, output_key='title', memory=idea_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)


if prompt:
    video_idea = idea_chain.run(prompt)
    script = script_chain.run(title=video_idea)

    st.write("### Video Idea:")
    st.write(video_idea)
    st.write("### Video Script:")
    st.write(script)

    favorite_content = f"Video Idea:\n{video_idea}\n\nVideo Script:\n{script}\n"
    download_file = io.BytesIO(favorite_content.encode())
    download_file.name = "favorite_tiktok_ideas.txt"

    st.download_button(
        label="Download Favorite Ideas",
        data=download_file,
        file_name="favorite_tiktok_ideas.txt",
        mime="text/plain"
    )

    with st.expander('Video Idea History'):
        st.info(idea_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)
