import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
models = client.models.list()
names = sorted([m.id for m in models.data])
print("\n".join(names))
print("\nHas text-embedding-3-small:", "text-embedding-3-small" in names)
