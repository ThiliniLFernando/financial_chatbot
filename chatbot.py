# chatbot.py
import openai

openai.api_key = 'your_openai_api_key'

def generate_financial_advice(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()