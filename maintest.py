# # INITIAL: Using OpenAI
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o", api_key="sk-...")
#
# # SWITCH: Using Local Model (via Ollama)
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2")
#
# # THE REST OF YOUR CODE STAYS THE SAME
# response = llm.invoke("Analyze this stock trend...")