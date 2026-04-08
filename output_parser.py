from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  # ✅ Fixed import
from langchain_core.output_parsers import StrOutputParser  # ✅ Fixed import
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
parser=StrOutputParser()

chain=template1 | model | parser | template2 |model | parser

result=chain.invoke({'topic':'black holes'})

print(result)

