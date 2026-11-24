from semantic_router import Route
from pathlib import Path
from typing import Any
import yaml
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router import Route, IndexedRouter

def load_config(config_path: Path | None = None) -> dict[str, Any]:
    resolved_path = Path("config.yaml")
    with resolved_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
config = load_config()


sql_route = Route(
    name="sql_route",
    utterances=[
        "How many tickets are currently open?",
        "How many critical-priority tickets exist?",
        "Which city has the most support tickets?",
        "Show all unresolved broadband complaints.",
        "How many tickets are assigned to Network Operations?",
        "Show all billing-related complaints from Bengaluru.",
        "Show me customer support ticket details.",
        "How many tickets were created today?",
        "Which team handles the most tickets?",
        "Show ticket statistics by city.",
        "Find tickets by customer ID.",
        "What is the status of a particular ticket?"
    ]
)

vector_route = Route(
    name="vector_route",
    utterances=[
        "What telecom plans are available?",
        "Explain the broadband installation process.",
        "What is the refund policy?",
        "How can I replace my SIM card?",
        "What documents are required for a new connection?",
        "Explain the cancellation policy.",
        "How can I troubleshoot network connectivity issues?",
        "What are the terms and conditions of my plan?",
        "How do I raise a customer complaint?",
        "Explain the company's service policies.",
        "What is the procedure for broadband activation?",
        "How can I contact customer support?"
    ]
)

encoder = HuggingFaceEncoder(name=config["ROUTING_EMBEDDING_MODEL"])

question_router = IndexedRouter(
    encoder=encoder,
    routes=[
        sql_route,vector_route
        ]
    )

def get_question_route(question: str):

    result = question_router(question)

    if result is None or result.name is None:
        return "unknown"
    print(f"From Router==============>{result.name}")
    return result.name

# if __name__ == "__main__":

#     questions = [
#         "Which city has the most support tickets?",
#         "How many critical tickets are open?",
#         "What is the refund policy?",
#         "How can I replace my SIM card?"
#     ]

#     for question in questions:

#         route_name = get_question_route(question)

#         print("\nQuestion:", question)
#         print("Selected Route:", route_name)