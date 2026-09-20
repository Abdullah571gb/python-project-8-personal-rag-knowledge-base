import re, time, datetime
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ─── KNOWLEDGE BASE CONTENT ────────────────────────

knowledge_documents = [
    Document(
        page_content="""Abdullah's AI Automation Learning Journey.
Started from zero with no programming knowledge.
Learning 6 hours per day across 24 weeks.
Goal: Become an AI automation expert and earn money.
Using only free tools throughout the entire journey.
Free tool stack: n8n, Groq API, Google Sheets, Telegram,
Notion, Google Colab, Ollama, GitHub, Hoppscotch.""",
        metadata={"source": "personal_info", "category": "background"}
    ),
    Document(
        page_content="""Completed learning: Phase 1 — AI Foundations.
Topics: How LLMs work, Prompt Engineering, 6-element formula.
Prompt techniques: Chain of Thought, JSON output, Few-shot,
Negative prompting, System vs User prompt separation.
Key insight: Strong prompts are the foundation of all automation.""",
        metadata={"source": "phase1", "category": "completed"}
    ),
    Document(
        page_content="""Completed learning: Phase 2 — n8n Automation.
Topics: Workflow basics, IF/Filter/Switch nodes, Loops,
Split in Batches, Error handling with master Error Catcher,
HTTP Request node for direct API calls, Schedulers and Cron,
Webhooks and triggers, Sub-workflows.
Projects built: Email AI Summarizer, Webhook Lead Capture,
Daily Briefing System (RSS + Groq + Notion + Telegram),
AI Lead Capture System.
Key insight: n8n is enough to build real business automation.""",
        metadata={"source": "phase2", "category": "completed"}
    ),
    Document(
        page_content="""Completed learning: Phase 3 — Python for AI Automation.
Topics: Python fundamentals, API calls with requests,
Groq API direct integration, Google Sheets with gspread,
Flask web server, ngrok for public URLs,
Python + n8n integration via webhooks.
Projects built: Lead Processor Pipeline (Sheets + AI + Telegram),
Python Flask API called by n8n, Complete Python + n8n system.
Key insight: Python adds power that n8n cannot provide alone.""",
        metadata={"source": "phase3", "category": "completed"}
    ),
    Document(
        page_content="""Current learning: Phase 4 — AI Agents.
Day 19: AI Agent concepts, ReAct framework, n8n AI Agent node,
adding tools (calculator, weather, news), memory systems.
Day 20: LangChain, ChatGroq, DuckDuckGo search, Wikipedia tools,
custom @tool decorated tools, ConversationBufferMemory.
Day 21: CrewAI, multi-agent teams, Agent/Task/Crew/Process,
content creation crew, lead research crew, agency proposal crew.
Day 22: RAG systems, vector stores, embeddings, FAISS,
conversational RAG, RAG + CrewAI combination.
Key insight: Agents think and act — automation just follows rules.""",
        metadata={"source": "phase4", "category": "current"}
    ),
    Document(
        page_content="""Coming next: Phase 5 — Expert Level.
Topics to cover: MCP (Model Context Protocol),
Production deployment of automation systems,
Advanced agent patterns and architectures,
Building real client projects,
Portfolio development and GitHub,
Freelancing with AI automation skills,
Earning money from AI automation.
Target: Complete 24-week journey and start getting clients.""",
        metadata={"source": "phase5", "category": "upcoming"}
    ),
    Document(
        page_content="""Projects portfolio built so far.
Project 1: Text → AI → Google Sheets (n8n basics).
Project 2: Gmail → AI Summary → Telegram (real triggers).
Project 3: Webhook → AI → Google Sheets (webhook processing).
Project 4: RSS News → AI → Notion (scheduled automation).
Project 5: Python Lead Processor (Python + Groq + Sheets).
Project 6: Python Data Pipeline (Sheets → AI → Sheets + Telegram).
Project 7: Python + n8n Integration (Flask API + webhooks).
Project 8: Personal AI Assistant (LangChain + tools + memory).
All projects documented on GitHub repository.""",
        metadata={"source": "portfolio", "category": "projects"}
    ),
    Document(
        page_content="""Key concepts mastered so far.
Automation: triggers, webhooks, polling, error handling, loops.
Prompt Engineering: 6 elements, chain of thought, JSON output.
Python: variables, functions, loops, API calls, error handling.
AI APIs: Groq, Gemini, direct HTTP calls, response parsing.
n8n Advanced: IF/Switch/Filter, Split in Batches, sub-workflows.
LangChain: agents, tools, memory, chains, prompt templates.
CrewAI: agents, tasks, crews, sequential process, custom tools.
RAG: document loading, chunking, embeddings, vector stores, retrieval.""",
        metadata={"source": "concepts", "category": "knowledge"}
    )
]

# ─── BUILD KNOWLEDGE BASE ──────────────────────────

print("Building personal knowledge base...")

kb_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
knowledge_chunks = kb_splitter.split_documents(knowledge_documents)
knowledge_store = FAISS.from_documents(knowledge_chunks, embeddings)
kb_retriever = knowledge_store.as_retriever(search_kwargs={"k": 4})

print(f"✅ Knowledge base built: {len(knowledge_chunks)} chunks")

# ─── CHAINS (replace ConversationalRetrievalChain) ─

# Step 1: turn follow-ups into standalone questions for better retrieval
kb_condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question as a standalone question using "
               "the chat history. If it is already standalone, return it unchanged. "
               "Return ONLY the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
kb_condense_chain = kb_condense_prompt | llm | StrOutputParser()

# Step 2: answer with context + history
kb_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Abdullah's personal AI learning assistant.
You have access to his complete learning journey documentation.

Use the context below to answer questions about his progress,
what he has learned, what projects he built, and what comes next.

If asked about something not in the knowledge base say:
"That is not in your learning documentation yet."

Always be encouraging and reference his specific achievements.

Context from knowledge base:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
kb_answer_chain = kb_answer_prompt | llm | StrOutputParser()

# ─── MEMORY (replaces ConversationBufferWindowMemory, k=8) ─

kb_history = []
KB_WINDOW = 8

# ─── RATE-LIMIT SAFE CALL (Groq free tier) ─────────

def call_with_retry(chain, payload, tries=6):
    for _ in range(tries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "Rate limit" in msg or "429" in msg:
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
                wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 2) if m else 15
                print(f"⏳ Rate limit, waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Still rate-limited after retries")

# ─── CHAT INTERFACE ────────────────────────────────

def ask_personal_assistant(question):
    print(f"\n👤 You: {question}")
    try:
        recent = kb_history[-2 * KB_WINDOW:]

        standalone = call_with_retry(
            kb_condense_chain, {"question": question, "chat_history": recent}
        ) if recent else question

        docs = kb_retriever.invoke(standalone)
        context = "\n\n".join(d.page_content for d in docs)

        answer = call_with_retry(kb_answer_chain, {
            "context": context,
            "question": question,
            "chat_history": recent
        })

        kb_history.append(HumanMessage(content=question))
        kb_history.append(AIMessage(content=answer))

        sources = sorted(set(d.metadata.get("source", "unknown") for d in docs))
        print(f"\n🤖 Assistant: {answer}")
        print(f"📚 From: {', '.join(sources)}")
        return answer
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ─── TEST THE ASSISTANT ────────────────────────────

print("\n" + "=" * 55)
print("🤖 PERSONAL KNOWLEDGE BASE ASSISTANT")
print(f"Loaded: {len(knowledge_documents)} knowledge documents")
print(f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}")
print("=" * 55)

ask_personal_assistant("What have I learned so far in my journey?")
ask_personal_assistant("What projects have I built?")
ask_personal_assistant("What phase am I currently in?")
ask_personal_assistant("What do I need to learn next?")
ask_personal_assistant("What tools do I use that are completely free?")
ask_personal_assistant("What did I just ask you about?")
ask_personal_assistant("Give me a motivational summary of my progress")