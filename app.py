import io
import os
import threading
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import pandas as pd
import requests
import yaml
from docling.chunking import HybridChunker
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from transformers import AutoTokenizer
from routing import get_question_route
from sql_rag import TicketSQLRAG
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

load_dotenv()


# Reads settings from config.yaml so model names, paths, or keys aren't hardcoded everywhere
def load_config(config_path: Path | None = None) -> dict[str, Any]:
    resolved_path = Path("config.yaml")
    with resolved_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Main helper class that loads TelecomCo knowledge once and answers questions using that knowledge
class TelecomRAGService:
    """Downloads, chunks, embeds, and indexes documents once when initialize() is called."""

    # Runs when creating a TelecomRAGService object; sets up initial blank/empty states
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.confs = config if config is not None else load_config()
        print("************")
        print(self.confs)
        self._lock = threading.Lock()  # Safety lock so setup doesn't collide if triggered twice at once
        self._initialized = False  # Tracks whether the heavy one-time setup has finished yet

        self.vectorstore: QdrantVectorStore | None = None  # Will hold the searchable database of documents
        self.retriever = None  # Will act as the "librarian" that finds the most relevant snippets
        self.chain = None  # Will hold the full pipeline: find info -> build prompt -> ask AI -> return answer
        self.sql_rag = TicketSQLRAG()

    # Quick check (True/False) telling callers whether the system is ready to answer questions
    @property
    def is_ready(self) -> bool:
        print("#@#@#@#@##@#@#@#@#@#@#@#@#@#@#@#@#@")
        print(self._initialized)
        print(self.chain)
        return self._initialized and self.chain is not None

    # Gathers knowledge from the FAQ CSV and PDF guide and turns them into searchable pieces
    def _get_documents(self) -> list[Document]:
        # 1. Process CSV FAQ
        csv_docs: list[Document] = []
        csv_url = "data/faq.csv"
        df_csv = pd.read_csv(csv_url)
        # Go through each FAQ row one by one and format it as a Question + Answer document
        for _, row in df_csv.iterrows():
            csv_docs.append(
                Document(
                    page_content=f"Question: {row['question']}\nAnswer: {row['answer']}",
                    metadata={
                        "source": "faq",
                        "category": row["category"],
                        "id": row["id"],
                    },
                )
            )

        # 2. Download and Process PDF
        # local_pdf_path = Path(self.confs.get("LOCAL_PDF_PATH", "/tmp/downloaded_document.pdf"))
        # local_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # response = requests.get(
        #     f"{self.confs['GITHUB_RAW_BASE']}/{self.confs['PDF_URL']}",
        #     timeout=60,
        # )
        # response.raise_for_status()
        # local_pdf_path.write_bytes(response.content)

        # PROJECT_ROOT = Path(__file__).resolve().parent.parent
        pdf_path = Path("data/downloaded_document.pdf")

        # Stop with a clear error if the PDF file isn't where expected
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown_output = result.document.export_to_markdown()  # Convert PDF layout into clean Markdown text

        # 3. Chunk PDF Content
        tokenizer = AutoTokenizer.from_pretrained(self.confs["EMBED_MODEL"])
        # Split the document into chunks up to 128 tokens and merge neighboring small pieces when appropriate
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=128, merge_peers=True)
        stream = DocumentStream(
            name="document.md",
            stream=io.BytesIO(markdown_output.encode("utf-8")),
        )
        md_converter = DocumentConverter(allowed_formats=[InputFormat.MD])
        md_result = md_converter.convert(stream)

        chunks = list(chunker.chunk(dl_doc=md_result.document))
        pdf_docs: list[Document] = []
        # Turn each PDF chunk into a Document and attach which section heading it came from
        for i, chunk in enumerate(chunks):
            headings = chunk.meta.headings if chunk.meta and chunk.meta.headings else []
            source_heading = headings[-1] if headings else "Unknown Section"
            pdf_docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source": "telecom_guide.pdf",
                        "chunk_id": i,
                        "section": source_heading,
                    },
                )
            )

        return csv_docs + pdf_docs  # Combine FAQ entries and PDF chunks into one list

    # Builds the search database (vectorstore), search tool (retriever), and AI answer pipeline (chain)
    def _build_chain(self, all_docs: list[Document]):
        # Meaning-based ("dense") search model that understands synonyms and concepts
        dense_embeddings = HuggingFaceEmbeddings(
            model_name=self.confs["EMBED_MODEL"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        print(f"**************************************dense_embeddings: {dense_embeddings}")
        # Keyword/exact-term ("sparse" BM25) search model good at matching specific terms or codes
        sparse_embeddings = FastEmbedSparse(
            model_name="Qdrant/bm25",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False, "batch_size": 32},
        )
        print(f"**************************************sparse_embeddings: {sparse_embeddings}")

        qdrant_path = self.confs.get("QDRANT_PATH", "/tmp/my_lang_vs2")
        # Create a hybrid search database combining meaning-based and keyword-based search
        vectorstore = QdrantVectorStore.from_documents(
            documents=all_docs,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            path=qdrant_path,
            collection_name=self.confs["COLLECTION_NAME"],
            retrieval_mode=RetrievalMode.HYBRID,
        )
        print(f"**************************************vectorstore: {vectorstore}")

        # Configure search to return up to 5 (k=5) matching snippets above a minimum relevance score (0.1)
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.1, "k": 5},
        )
        print(f"**************************************retriever: {retriever}")
        # Look up the Groq API password/key from the environment or config file; fail early if missing
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables and config.yaml")

        # Set up the AI model with temperature=0 so answers stay factual and consistent rather than creative
        llm = ChatGroq(
            model=self.confs["GROQ_MODEL"],
            api_key=groq_api_key,
            temperature=0,
        )

        # Rules telling the AI to stick strictly to retrieved company info and admit when unsure
        system_prompt = """You are a helpful TelecomCo customer support assistant.
                            Answer the customer's question using ONLY the information provided in the context below.
                            If the answer is not in the context, say "I don't have that information."
                            Keep answers concise and friendly.

Context:
{context}"""

        # Combine the system rules and the user's question into a structured chat prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # Join multiple retrieved text snippets together with blank lines between them
        def _format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # Connect the steps in order: fetch matching docs & keep question -> fill prompt -> ask AI -> get text
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        return vectorstore, retriever, chain

    # Runs the full startup preparation once (unless force=True is passed to rebuild)
    def initialize(self, force: bool = False) -> None:
        """Runs once at startup to load docs, embed chunks, and build the chain."""
        if self._initialized and not force:
            return

        # Only one thread can run this block at a time
        with self._lock:
            if self._initialized and not force:
                return
            all_docs = self._get_documents()
            print(f"**************************************all_docs")
            print(all_docs[0])
            # Try building the search index and AI chain, printing an error message if something fails
            try:
                self.vectorstore, self.retriever, self.chain = self._build_chain(all_docs)
            except Exception as e:
                print(f"Error occurred while building chain: {e}")
            self._initialized = True

    # Returns just the matching document snippets for a question without calling the AI to write an answer
    def retrieve(self, question: str) -> list[Document]:
        """Real-time retrieval only."""
        if not self.is_ready:
            raise RuntimeError("RAG service is not initialized.")
        return self.retriever.invoke(question)


    # Finds matching document snippets for a question and asks the AI to write a helpful answer
    def ask(self, question: str) -> str:
        """Real-time retrieval + LLM answer generation."""
        print("HIiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        if not self.is_ready:
            raise RuntimeError("RAG service is not initialized.")
        # Step 1: Identify the question route
        route_name = get_question_route(question)
        print(f"Question==========>{question}Route Name=============================================>{route_name}")

        # Step 2: SQL RAG
        if route_name == "sql_route":
            print("Selected Route: sql_route")
            response,sql = self.sql_rag.ask(question)
            return {"answer":response,"route":route_name,"Query":sql}

        # Step 3: Vector RAG
        elif route_name == "vector_route":
            print("Comes inside vector===================")
            response = self.chain.invoke(question)
            return {"answer":response,"route":route_name,"Query":"NA"}


# Shortcut function that creates the service, runs setup, and returns just the answer chain
def build_and_get_chain() -> Any:
    """Optional helper if any legacy caller still expects a chain directly."""
    service = TelecomRAGService()
    service.initialize()
    return service.chain