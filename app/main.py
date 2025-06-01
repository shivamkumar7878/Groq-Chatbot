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
from langchain.chat_models import init_chat_model  # Correct import
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class FetchData(BaseModel):
    """Extracts an SQL query from natural language input."""

    query: str = Field(..., description="The generated SQL query.")


system_prompt = """

You are a helpful assistant designed to answer user questions by querying a database.  You have access to a tool called `FetchData`, to fetch data from the SQL database.

The database contains a single table named `customers`, defined as:
>  CREATE  TABLE customers (
>     customer_id INTEGER  PRIMARY KEY,
>     name VARCHAR,
>     gender VARCHAR,
>     location VARCHAR );

When filtering or comparing values in the `gender` or `location` columns, use `LOWER()` to ensure case-insensitive matching.

Your job is to:
1.  Understand the user's question.
2.  If the answer requires querying the database, use the `FetchDAta` tool to extract relevant data from database.
3.  Use the result of the query to answer the user's question in a clear and helpful way.
4.  Final answer should be in the form of a natural language response, not SQL code. do not include the tool usage in the final answers."""


async def get_sql_from_llm(prompt: str) -> str:
    """Fallback method using direct API call."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
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
        # Initialize the chat model - correct model name for Groq
        llm = init_chat_model("llama3-8b-8192", model_provider="groq")

        # Bind the SQLQueryExtractor tool
        llm_with_tools = llm.bind_tools([FetchData])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=natural_query),
        ]
        for i in range(3):
            ai_msg = llm_with_tools.invoke(messages)
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "FetchData":
                        sql_query = tool_call["args"].get("query")
                        data = await get_sql_from_llm(sql_query)
                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content=data,
                            )
                        )
            else:
                # logging.info("No tool calls found, retrying...", ai_msg.content)
                print(f"No tool calls found, retrying... {ai_msg.content}")
                print(f"AI Message: {ai_msg}")
                return {"result": ai_msg.content}

        logging.info(f"AI Message type: {type(ai_msg)}")
        logging.info(
            f"AI Message content: {ai_msg.content if hasattr(ai_msg, 'content') else 'No content'}"
        )
        logging.info(
            f"AI Message tool_calls: {ai_msg.tool_calls if hasattr(ai_msg, 'tool_calls') else 'No tool_calls'}"
        )

        if not sql_query:
            logging.warning("No valid SQL from tool call, using fallback method")
            if tool_error_details:
                logging.info(f"Tool error was: {tool_error_details}")
            return ai_msg.content

        db: Session = next(get_db())
        result = db.execute(text(sql_query)).fetchall()
        logging.info(f"Query executed successfully, fetched {len(result)} rows.")
        logging.debug(f"Query Result: {result}")

        return {"results": [dict(row._mapping) for row in result]}
    except Exception as tool_error:
        tool_error_details = str(tool_error)
        logging.error(f"Tool calling failed with error: {tool_error}")
        logging.error(f"Tool error type: {type(tool_error)}")
        import traceback

        logging.error(f"Full traceback: {traceback.format_exc()}")


@app.get("/")
async def root():
    return {"message": "Customer Query API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
