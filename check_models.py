import openai
from config import OPENAI_API_KEY

''' 
This script lists all the models available in your OpenAI account.
'''

client = openai.Client(api_key=OPENAI_API_KEY)
models = client.models.list()

for model in models:
    print(model.id)
