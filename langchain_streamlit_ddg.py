# langchain with duckduckgo tool + streamlit GUI
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
import os

# Initialize session state for chat history if not exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for API key input
openai_api_key = st.sidebar.text_input("Qwen API Key")

# Create memory for the conversation
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    human_prefix="Human",
    ai_prefix="Assistant"
)

# Display chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role.lower()).write(message)
'''
This will now show the entire conversation history, with each message properly formatted as either a user or assistant message.
The history persists across interactions as long as the Streamlit session is active.

Streamlit has a unique way of handling page updates through what they call "reactive rerunning." Here's how it works:
Common Refresh Triggers
 - User interaction with any widget
 - Code changes in the script
 - Using st.rerun() explicitly
 - Using st.experimental_rerun() (older versions)
 - Browser page refresh

Caching Mechanisms
Streamlit uses caching to avoid expensive recomputations
@st.cache_data - Caches the output of functions returning data
@st.cache_resource - Caches resources like database connections or ML models
Session state (st.session_state) persists data between reruns

'''
    
if prompt := st.chat_input("What is up?"):
    # The walrus operator := assigns the result of st.chat_input() to the variable prompt AND 
    # evaluates the truthiness of that value in the if statement at the same time. 
    # It's equivalent to writing:
    # prompt = st.chat_input("What is up?")
    # if prompt:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    # Create a LangChain compatible OpenAI chat model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus"  # Specify the model
    )
    
    # Load tools
    tools = load_tools(['ddg-search'])
    
    # Create agent with memory
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,  # Add memory to the agent
    )
    
    # Add user message to chat history and display it
    st.session_state.chat_history.append(("Human", prompt))
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(
            input=prompt, 
            chat_history=st.session_state.chat_history,
            callbacks=[st_callback]
        )
        # The StreamlitCallbackHandler is used to display the agent's thought process in real-time in the Streamlit UI. When you use:
        '''
        This creates a more transparent and interactive experience as users can see how the agent:
         - Thinks about the problem
         - Chooses which tools to use
         - Processes the information
         - Arrives at its final response
        It's especially useful for debugging and helping users understand how the agent reached its conclusions rather than just seeing the final output.
        Without the callback, you would only see the final response without any insight into the agent's decision-making process.
        '''
        
        # Add assistant response to chat history and display it
        st.session_state.chat_history.append(("Assistant", response))
        st.write(response)
        