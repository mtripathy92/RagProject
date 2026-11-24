import os
from pathlib import Path
import sqlite3
from typing import Any
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
import re

import yaml

def load_config(config_path: Path | None = None) -> dict[str, Any]:
    resolved_path = Path("config.yaml")
    with resolved_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

"""
"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"
"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"
"""

class TiscketSQLRAG:

    def __init__(self):        
        self.confs = load_config()
        self.db_path = "data/telecom_support.db"
        self.conn = sqlite3.connect(self.db_path)
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        # tables_df = pd.read_sql("SELECT * FROM support_tickets limit 5;", conn)
        # print(tables_df)

        # Initialize LLm
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables and config.yaml")

        self.llm = ChatGroq(
                    model=self.confs["GROQ_MODEL"],
                    api_key=groq_api_key,
                    temperature=0,
                )
        if self.llm is None:
            print("LLM Creation failed")
        
        self.sql_query_chain = create_sql_query_chain(self.llm,self.db)
        if self.sql_query_chain is None:
            print("SQL Query chain creation failed")


    def clean_sql(self, sql_query: str) -> str:

        # Remove leading/trailing whitespace
        sql_query = sql_query.strip()

        # Remove Markdown code fences
        sql_query = re.sub(
            r"^```(?:sql)?\s*",
            "",
            sql_query,
            flags=re.IGNORECASE
        )

        sql_query = re.sub(
            r"\s*```$",
            "",
            sql_query
        )

        # Remove SQL label if present
        sql_query = re.sub(
            r"^SQL\s*:\s*",
            "",
            sql_query,
            flags=re.IGNORECASE
        )

        # Extract query if LLM includes an explanation
        match = re.search(
            r"\b(SELECT|WITH)\b.*",
            sql_query,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not match:
            raise ValueError("No valid SQL query found in LLM response")

        sql_query = match.group(0).strip()

        # Remove trailing semicolons
        sql_query = sql_query.rstrip(";").strip()

        return sql_query     
                                                              
    def sql_rag_chain(self, question: str) -> str:
        # Step 1: Generate SQL from the natural language question
        raw_sql = self.sql_query_chain.invoke({"question": question})
        sql = self.clean_sql(raw_sql)
        print(f"[debug] cleaned SQL → {sql}")

        # Step 2: Execute the SQL against the database
        result = self.db.run(sql)
        SYSTEM_PROMPT = """You are a TelecomCo support analytics assistant.
Given a user question and the SQL query result from our tickets database,
provide a clear, concise natural language answer.
Be specific with numbers and facts from the data."""
        # Step 3: Ask the LLM to turn the raw result into a natural language answer
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\nSQL Result: {result}\n\nAnswer:"),
        ])
        response = answer_prompt | self.llm
        return response.invoke({"question": question, "result": result}).content,sql       
    

    def ask(self, question):
        answer,sql = self.sql_rag_chain(question)
        return answer,sql