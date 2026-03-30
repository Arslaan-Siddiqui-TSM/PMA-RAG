from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a project management assistant that answers questions "
            "strictly based on the provided document context.\n\n"
            "Rules:\n"
            "- Only answer based on the provided context\n"
            "- If the context does not contain enough information, say: "
            "\"I don't have enough information in the documents to answer "
            'this question."\n'
            "- Be precise and factual",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

INTENT_CLASSIFY_PROMPT = """\
You are an intent classifier for a project management document assistant.
Classify the user's message into exactly ONE category. Reply with ONLY the category name.

Categories:
- doc_query: User is asking a question about project documents (PRDs, BRDs, specs, test plans, etc.)
- followup: User is asking a follow-up to a previous answer (e.g. "tell me more", "what about X?", "can you elaborate?", references to "that", "it", "those")
- comparison: User wants to compare information across different documents or document types (e.g. "compare BRD and PRD", "differences between functional and non-functional")
- summary: User wants a summary of a specific document or document type (e.g. "summarize the test plan", "give me an overview of the BRD")

Recent conversation:
{chat_history}

User message: {question}
Category:"""

REFORMULATE_PROMPT = """\
You are a query reformulator. Given the conversation history and the user's \
latest message, produce a single standalone question that captures the user's \
full intent without requiring any prior context.

Also determine: does this follow-up ask about the SAME topic as the previous \
answer, or a NEW topic?
- If SAME topic: reply with REUSE: followed by the standalone question
- If NEW topic: reply with RETRIEVE: followed by the standalone question

Conversation history:
{chat_history}

User's latest message: {question}

Reply with either "REUSE: <question>" or "RETRIEVE: <question>":"""

CASUAL_RESPONSES = {
    "greeting": (
        "Hi there! I'm your project management document assistant. "
        "I can help you query, compare, and summarize your project documents "
        "(PRDs, BRDs, specs, test plans, and more).\n\n"
        "Upload some documents or ask me a question to get started!"
    ),
    "thanks_bye": (
        "You're welcome! Feel free to ask if you have more questions "
        "about your project documents. I'm here to help!"
    ),
}

HELP_RESPONSE = (
    "Here's what I can do:\n\n"
    "**Document Queries**\n"
    "Ask questions about your uploaded documents. "
    'Example: *"What are the functional requirements?"*\n\n'
    "**Follow-up Questions**\n"
    "Ask follow-ups and I'll use our conversation context. "
    'Example: *"Tell me more about that"* or *"What about section 3?"*\n\n'
    "**Compare Documents**\n"
    "Compare information across different document types. "
    'Example: *"Compare the BRD and PRD requirements"*\n\n'
    "**Summarize Documents**\n"
    "Get summaries of specific documents or types. "
    'Example: *"Summarize the test plan"*\n\n'
    "**Upload Documents**\n"
    "Use the file upload button or type `/upload` to add new documents "
    "(PDF, DOCX, or Markdown).\n\n"
    "**Filter by Type**\n"
    "Use the settings panel to filter queries by document type."
)
