from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  # ✅ Fixed import

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")  # ✅ Pass model name
template1 = PromptTemplate(  # ✅ Fixed class name
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(  # ✅ Fixed class name
    template='write a 5 line summary on the following text. \n{text}',  # ✅ Fixed \n
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black holes'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})  # ✅ Fixed variable reference
result1 = model.invoke(prompt2)

print(result1.content)