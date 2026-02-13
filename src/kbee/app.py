"""Chainlit chat UI for KBee.

Provides an interactive chat interface where users can ask questions
and receive AI-generated answers based on the knowledge base.
"""

import re

import chainlit as cl

from kbee.query import get_query_engine, query_knowledge_base


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the query engine and send a welcome message."""
    try:
        query_engine = get_query_engine()
        cl.user_session.set("query_engine", query_engine)
        await cl.Message(
            content=(
                "Hi，我是 KBee。請問我可以怎麼協助您呢。"
            ),
        ).send()
    except FileNotFoundError:
        await cl.Message(
            content=(
                "⚠️ Knowledge base not found. "
                "Please run ingestion first:\n\n"
                "```bash\nuv run python -m kbee.ingest\n```"
            ),
        ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages and respond with RAG answers."""
    query_engine = cl.user_session.get("query_engine")
    if query_engine is None:
        await cl.Message(
            content="⚠️ Knowledge base is not loaded. Please restart the chat.",
        ).send()
        return

    # Show thinking indicator.
    msg = cl.Message(content="")
    await msg.send()

    if _is_greeting(message.content):
        msg.content = "您好，我在這裡。請告訴我您想詢問的問題，我會盡快協助您。"
        await msg.update()
        return

    # Query the knowledge base.
    response = await query_knowledge_base(query_engine, message.content)

    # Format the response with source references.
    answer = str(response)
    sources = _format_sources(response)

    if sources:
        answer += f"\n\n---\n📚 **Sources:**\n{sources}"

    msg.content = answer
    await msg.update()


def _format_sources(response) -> str:
    """Format source node references from the query response.

    Args:
        response: The LlamaIndex query response.

    Returns:
        A formatted string listing source documents and their
        relevance scores.
    """
    if not hasattr(response, "source_nodes") or not response.source_nodes:
        return ""

    lines = []
    for i, node in enumerate(response.source_nodes, 1):
        filename = node.metadata.get("file_name", "Unknown")
        score = node.score if node.score is not None else 0.0
        preview = node.text[:120].replace("\n", " ").strip()
        lines.append(
            f"- **[{i}] {filename}** (relevance: {score:.2f})\n  _{preview}..._"
        )

    return "\n".join(lines)


def _is_greeting(text: str) -> bool:
    """Return True if the message is a simple greeting."""
    compact = re.sub(r"\s+", "", text.strip().lower())
    greeting_set = {
        "hi",
        "hikbee",
        "hello",
        "hellokbee",
        "hey",
        "heykbee",
        "嗨",
        "嗨kbee",
        "哈囉",
        "哈囉kbee",
        "你好",
        "你好kbee",
        "您好",
        "您好kbee",
        "安安",
        "安安kbee",
    }
    return compact in greeting_set
