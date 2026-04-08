from click import prompt

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='generate a deailed report on {topic}',
    input_variables=['topic']
)   

prompt2 = PromptTemplate(
    template='write a 5 line summary on the following text. \n{text}',
    input_variables=['text']
)
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'unemployement in india'})

print(result)

chain.get_graph().print_ascii()