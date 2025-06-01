from fastapi import FastAPI, HTTPException, Request
from sqlalchemy.orm import Session
from database import SessionLocal, Customer
from dotenv import load_dotenv
import os
import httpx
import logging
import re
from sqlalchemy import text
import getpass
from langchain.chat_models import init_chat_model


load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# os.environ["GROQ_API_KEY"] = getpass.getpass(GROQ_API_KEY)


# GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"  # Adjust if different


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from pydantic import BaseModel, Field


class SQLQueryExtractor(BaseModel):
    """Extracts an SQL query from natural language input."""

    query: str = Field(..., description="The generated SQL query.")


async def get_sql_from_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that converts natural language to SQL. "
                    "Use the table 'customers'. It was created with: "
                    "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR, gender VARCHAR, location VARCHAR); "
                    "Always compare values of 'gender' and 'location' using LOWER() for case-insensitive matching."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GROQ_URL, json=payload, headers=headers)

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@app.post("/query/")
async def query(request: Request):
    data = await request.json()
    natural_query = data.get("query")
    if not natural_query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # Step 1: Get SQL from LLM
        llm_response = await get_sql_from_llm(natural_query)

        # llm = init_chat_model("llama3-8b-8192", model_provider="groq")
        # llm_with_tools = llm.bind_tools([SQLQueryExtractor])

        # ai_msg = llm_with_tools.invoke(natural_query)
        # response = ai_msg.tool_calls

        # logging.info(f"AI Message: {ai_msg}")

        # if ai_msg.tool_calls:
        #     logging.info(f"Tool Calls: {ai_msg.tool_calls}")
        #     tool_call = ai_msg.tool_calls[0]
        #     if tool_call['name'] == 'SQLQueryExtractor':
        #         response = tool_call["args"].get("query", None)
        #         llm_response = await get_sql_from_llm(natural_query)

        # else:
        #     logging.warning("No tool calls found in AI message.")
        #     return ai_msg.content

        logging.info(f"Raw LLM Response:\n{llm_response}")

        # Step 2: Extract actual SQL from markdown
        sql_match = re.search(r"```sql\n(.*?)```", llm_response, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            sql_query = llm_response.strip()

        logging.info(f"Extracted SQL: {sql_query}")

        # Step 3: Execute SQL safely
        db: Session = next(get_db())
        result = db.execute(text(sql_query)).fetchall()
        logging.info(f"Query executed successfully, fetched {len(result)} rows.")
        logging.debug(f"Query Result: {result}")

        return {"results": [dict(row._mapping) for row in result]}

    except Exception as e:
        logging.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))
