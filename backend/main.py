import base64
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load .env
load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL_NAME")


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


llm = ChatOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)

image_path = "images.jpg"
image_base64 = encode_image(image_path)

prompt_text = """
You are an AI that classify image into labels, Please give 3 classification name for the image
"""

message = HumanMessage(
    content=[
        {"type": "text", "text": prompt_text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpg;base64,{image_base64}"},
        },
    ]
)

response = llm.invoke([message])
print("Classification:", response.content)
