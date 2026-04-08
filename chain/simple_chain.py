from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate          # ✅ Fix 1: PromptTemplate (camelCase)
from langchain_core.output_parsers import StrOutputParser  # ✅ Fix 2: langchain_core (typo was langchin)

load_dotenv()

prompt = PromptTemplate(                                   # ✅ Fix 3: PromptTemplate (camelCase)
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'black cricket'})
print(result)

chain.get_graph().print_ascii()