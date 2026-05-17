import base64
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

load_dotenv(override=True)

from schema import DetectionResult

# Initialize Gemini with structured output so the response is automatically
# parsed into a DetectionResult object.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
    temperature=0,
).with_structured_output(DetectionResult)

# Prompt template that accepts a base64-encoded image and a text query.
prompt = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template([
        {
            "image_url": {"url": "data:image/jpeg;base64,{image}"}
        },
        {
            "text": """\
Describe the image.
Then detect all objects matching "{query}".
For each object provide a label and box_2d ([y_min, x_min, y_max, x_max] on a 0-1000 scale).
"""
        },
    ])
])

# LCEL pipeline: encode image → build prompt → call LLM → return structured output
chain = (
    {
        "image": lambda x: base64.b64encode(x["image"]).decode(),
        "query": lambda x: x["query"],
    }
    | prompt
    | llm
)
