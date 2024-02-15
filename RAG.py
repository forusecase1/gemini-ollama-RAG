from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os

if "history" not in st.session_state:
    st.session_state.history = []

app = Flask(__name__)

load_dotenv()

# domain root
@app.route('/')
def home():
    return 'Hello, World!'

@app.route("/webhook", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

model_type= 'gemini'

# Initializing Gemini
if(model_type == "ollama"):
    model = Ollama(
                    model=<MODEL_NAME>,  # Provide your ollama model name here
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler])
                )
    
elif(model_type == "gemini"):
    model = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.1, 
                                convert_system_message_to_human=True
                            )
ine_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
line_handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
working_status = os.getenv("DEFALUT_TALKING", default = "true").lower() == "true"    
genai.configure(api_key=os.getenv["GOOGLE_API_KEY"])

# Vector Database
persist_directory = "./db/gemini" # Persist directory path
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(persist_directory):
    with st.spinner('🚀 Starting your bot.  This might take a while'):
        # Data Pre-processing
        pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)
        
        pdf_documents = pdf_loader.load()
        text_documents = text_loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)

        data = pdfs + texts

        print("Data Processing Complete")

        vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        print("Vector DB Creating Complete\n")

elif os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)
    
    print("Vector DB Loaded\n")

# Quering Model
query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever()
)

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


prompt = st.chat_input("Say something")
if prompt:
    st.session_state.history.append({
        'role':'user',
        'content':prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        response = query_chain({"query": prompt})

        st.session_state.history.append({
            'role' : 'Assistant',
            'content' : response['result']
        })

        with st.chat_message("Assistant"):
            st.markdown(response['result'])

@line_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    global working_status
    if event.message.type != "text":
        return

    if working_status:
        chatgpt.add_msg(f"HUMAN:{event.message.text}?\n")
        reply_msg =  st.markdown(response['result'])
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_msg))

if __name__ == "__main__":
    app.run()


