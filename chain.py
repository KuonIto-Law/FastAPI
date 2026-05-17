import base64
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

load_dotenv(override=True)

from schema import DetectionResult

# ① LLMの初期化（with_structured_outputでスキーマを直接渡す）
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
    temperature=0,
).with_structured_output(DetectionResult)

# ② プロンプト：画像（base64）+ テキスト指示
prompt = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template([
        {
            "image_url": {"url": "data:image/jpeg;base64,{image}"}
        },
        {
            "text": """\
画像を説明してください。
次に、画像の中から「{query}」に該当するものをすべて検出してください。
それぞれに label と box_2d（0〜1000スケールの [y_min, x_min, y_max, x_max]）を付けてください。
"""
        },
    ])
])

# ③ Chain の組み立て
chain = (
    {
        "image": lambda x: base64.b64encode(x["image"]).decode(),
        "query": lambda x: x["query"],
    }
    | prompt
    | llm
)
