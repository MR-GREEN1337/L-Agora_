import os


def load_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['MISTRAL_API_KEY'] = "Gp7WkrvFiaQ21ZF6XcnQl5jUiqzgNKEN"
    os.environ["GROQ_API_KEY"] = "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"
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["TAVILY_API_KEY"] = "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"
    # Optional, add tracing in LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "L'Agora"
    #print("Added params successfully")