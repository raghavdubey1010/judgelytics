# JUDGELYTICS - FastAPI Backend: Chat Router
# Purpose: Legal assistant chat endpoints powered by Groq Llama 3
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Chat endpoints for Judgelytics backend.

Provides AI-powered legal assistant using Groq Llama 3 (free tier).
Falls back to rule-based responses if Groq key is not configured.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from ..models.user import User
from ..models.chat import ChatMessage
from ..database import get_db
from ..core.security import get_current_user
from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Chat"])

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Nyaya, an AI legal assistant specialized in Indian consumer law.
You help users understand their consumer rights, legal remedies, and court procedures.

Your expertise covers:
- Consumer Protection Act 2019 (COPRA)
- Bharatiya Nyaya Sanhita (BNS) — replaced IPC in 2023
- District/State/National Consumer Disputes Redressal Commissions (DCDRC/SCDRC/NCDRC)
- E-commerce consumer rights (Consumer Protection (E-Commerce) Rules 2020)
- Product liability, deficiency of service, unfair trade practices
- Filing procedures, required documents, and timelines
- ADR options: mediation, Lok Adalat

Key COPRA 2019 facts you know:
- District Commission: claims up to ₹1 crore
- State Commission: claims ₹1–10 crores  
- National Commission: claims above ₹10 crores
- Limitation period: 2 years from cause of action
- Filing fee: District ₹200-1000, State ₹2500, National ₹5000-7500

Always:
- Give practical, actionable advice
- Mention specific sections/provisions when relevant
- Recommend consulting a lawyer for complex matters
- Be empathetic — users are dealing with grievances
- Respond in English but be clear and simple

Do not give advice on criminal matters, divorce, property disputes outside consumer context, or taxation.
Keep responses concise (3-5 paragraphs max). Use bullet points for lists.

IMPORTANT: When mentioning a specific Act, Rule, or Section (e.g., Section 35, COPRA 2019, Consumer Protection (E-Commerce) Rules 2020), you MUST format it as a markdown hyperlink pointing to the legal library search page using the 'q' parameter.
Example formats: 
- `[Section 35](legal-library.html?q=Section+35)`
- `[Consumer Protection Act 2019](legal-library.html?q=COPRA+2019)`
- `[Bharatiya Nyaya Sanhita](legal-library.html?q=BNS)`"""


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _rule_based_response(message: str) -> str:
    """
    Rule-based fallback legal responses when Groq is unavailable.
    """
    msg = message.lower()

    if any(w in msg for w in ["file", "how to", "process", "procedure", "complaint"]):
        return (
            "**How to File a Consumer Complaint:**\n\n"
            "1. **Identify the right forum** based on your claim amount:\n"
            "   - Up to ₹1 crore → District Consumer Commission\n"
            "   - ₹1-10 crore → State Consumer Commission\n"
            "   - Above ₹10 crore → National Consumer Commission (NCDRC)\n\n"
            "2. **Gather documents**: Purchase receipt, warranty, correspondence with company, "
            "defective product photos, medical reports (if applicable).\n\n"
            "3. **Send a legal notice** to the opposite party (optional but recommended).\n\n"
            "4. **File the complaint** at the commission office or online at "
            "[edaakhil.nic.in](https://edaakhil.nic.in) with prescribed filing fee.\n\n"
            "5. **Attend hearings** as scheduled. Cases typically resolve in 3-5 months.\n\n"
            "*You can also use this platform's **Case Analysis** feature to get a prediction of your case outcome!*"
        )

    elif any(w in msg for w in ["fee", "cost", "charge", "money", "fees"]):
        return (
            "**Consumer Court Filing Fees (COPRA 2019):**\n\n"
            "| Forum | Claim Amount | Filing Fee |\n"
            "|-------|-------------|------------|\n"
            "| District Commission | Up to ₹5 lakh | ₹200 |\n"
            "| District Commission | ₹5L - ₹10L | ₹400 |\n"
            "| District Commission | ₹10L - ₹20L | ₹500 |\n"
            "| District Commission | ₹20L - ₹50L | ₹2,000 |\n"
            "| District Commission | ₹50L - ₹1 crore | ₹4,000 |\n"
            "| State Commission | ₹1Cr - ₹10Cr | ₹5,000 |\n"
            "| National Commission | Above ₹10Cr | ₹7,500 |\n\n"
            "No fee is charged if you are a Below Poverty Line (BPL) complainant."
        )

    elif any(w in msg for w in ["document", "evidence", "proof", "paper"]):
        return (
            "**Documents Required for Consumer Complaint:**\n\n"
            "📄 **Mandatory:**\n"
            "- Purchase receipt / invoice / order confirmation\n"
            "- Identity proof (Aadhaar / PAN)\n"
            "- Written complaint (in prescribed format)\n\n"
            "📋 **Supporting Evidence:**\n"
            "- Warranty card / guarantee certificate\n"
            "- Photos/videos of defective product\n"
            "- All email/WhatsApp correspondence with the company\n"
            "- Legal notice (if sent) and proof of delivery\n"
            "- Medical records (for health-related complaints)\n"
            "- Bank statements (for financial fraud)\n\n"
            "💡 **Tip:** File all documents in duplicate — one set for the court, one for opposite party."
        )

    elif any(w in msg for w in ["time", "limit", "deadline", "days", "duration", "how long"]):
        return (
            "**Time Limits in Consumer Cases:**\n\n"
            "⏰ **Limitation Period:** File your complaint within **2 years** from the date the grievance arose "
            "(Section 69, COPRA 2019). Delay can be condoned with sufficient reason.\n\n"
            "📅 **Case Resolution Timelines:**\n"
            "- Simple cases: 3-5 months\n"
            "- Complex cases: 1-2 years\n"
            "- NCDRC matters: 2-3 years\n\n"
            "🚀 **Fast Track:** Online filing via edaakhil.nic.in and video conferencing hearings "
            "have significantly reduced timelines since 2021."
        )

    elif any(w in msg for w in ["ecommerce", "amazon", "flipkart", "online", "delivery", "refund"]):
        return (
            "**E-Commerce Consumer Rights (India):**\n\n"
            "Under the [**Consumer Protection (E-Commerce) Rules 2020**](legal-library.html?q=E-Commerce+Rules), you have the right to:\n\n"
            "✅ Full disclosure of seller information\n"
            "✅ Genuine product reviews and ratings\n"
            "✅ Grievance redressal within **48 hours** of complaint registration\n"
            "✅ Resolution within **1 month**\n"
            "✅ Refund for defective/wrong/damaged products\n\n"
            "**If the platform refuses a refund:**\n"
            "1. File a complaint on the National Consumer Helpline: **1915** (toll-free)\n"
            "2. File on the e-commerce platform's grievance cell\n"
            "3. File a case with District Consumer Commission\n\n"
            "Platform liability: Under COPRA, both the seller AND the marketplace can be held liable."
        )

    elif any(w in msg for w in ["insurance", "claim", "policy", "premium", "reject"]):
        return (
            "**Insurance Consumer Rights:**\n\n"
            "If your insurance claim is wrongly denied or delayed, you have multiple remedies:\n\n"
            "1. **Internal Grievance**: File with the insurer's Grievance Cell (resolve within 15 days)\n"
            "2. **Insurance Ombudsman**: File with the nearest Insurance Ombudsman (free, fast)\n"
            "3. **IRDAI Bima Bharosa**: File on IRDAI's Bima Bharosa portal\n"
            "4. **Consumer Commission**: File a case for 'deficiency of service'\n\n"
            "**Key sections:** Section 45 (repudiation), Section 64VB (insurance policies)\n\n"
            "💡 Insurance disputes are well-suited for consumer court as insurers are frequently held liable "
            "for arbitrary claim rejection."
        )

    elif any(w in msg for w in ["bank", "fraud", "atm", "upi", "payment", "loan"]):
        return (
            "**Banking & Financial Consumer Rights:**\n\n"
            "For banking fraud or deficiency of service:\n\n"
            "🏦 **Immediate Steps:**\n"
            "- Report unauthorized transaction within **3 days** to limit your liability to zero\n"
            "- File FIR with police (cybercrime) at cybercrime.gov.in\n"
            "- File complaint with your bank branch and get acknowledgement\n\n"
            "📋 **Escalation Path:**\n"
            "1. Bank's Nodal Officer / Grievance Cell (15 days)\n"
            "2. **Banking Ombudsman** (RBI) — free, no lawyer needed\n"
            "3. **Consumer Commission** — for 'deficiency of service'\n\n"
            "**RBI Banking Ombudsman:** cms.rbi.org.in (online filing available)"
        )

    elif any(w in msg for w in ["copra", "consumer protection", "2019"]):
        return (
            "**Consumer Protection Act 2019 (COPRA):**\n\n"
            "[**COPRA 2019**](legal-library.html?q=COPRA) is the primary law protecting consumer rights in India. Key highlights:\n\n"
            "🏛️ **Three-Tier Dispute System:**\n"
            "- **District Consumer Commission**: Claims up to ₹1 crore\n"
            "- **State Consumer Commission**: Claims ₹1–10 crores\n"
            "- **National Consumer Commission (NCDRC)**: Claims above ₹10 crores\n\n"
            "📋 **Your Key Rights under COPRA 2019:**\n"
            "- Right to Safety\n"
            "- Right to Information\n"
            "- Right to Choose\n"
            "- Right to be Heard\n"
            "- Right to Redressal\n"
            "- Right to Consumer Education\n\n"
            "⏰ **Limitation Period**: 2 years from cause of action (Section 69)\n\n"
            "🆕 **New in 2019**: Product liability, e-commerce regulation, "
            "Central Consumer Protection Authority (CCPA), and mediation as ADR."
        )

    elif any(w in msg for w in ["section 35", "section 2", "section 47", "section 58", "section 69", "section 73", "section", "ipc", "bns", "provision"]):
        return (
            "**Key COPRA 2019 Sections for Consumer Cases:**\n\n"
            "📜 [**Section 2(7)**](legal-library.html?q=Section+2) — Definition of 'Consumer' (who can file)\n"
            "📜 [**Section 2(11)**](legal-library.html?q=Section+2) — Definition of 'Deficiency of Service'\n"
            "📜 [**Section 2(21)**](legal-library.html?q=Section+2) — Definition of 'Unfair Trade Practice'\n"
            "📜 [**Section 35**](legal-library.html?q=Section+35) — Complaints before District Consumer Commission\n"
            "📜 [**Section 47**](legal-library.html?q=Section+47) — Jurisdiction of State Consumer Commission\n"
            "📜 [**Section 58**](legal-library.html?q=Section+58) — Jurisdiction of National Consumer Commission\n"
            "📜 [**Section 69**](legal-library.html?q=Section+69) — Limitation period (2 years to file)\n"
            "📜 [**Section 73**](legal-library.html?q=Section+73) — Relief the commission can grant\n"
            "📜 [**Section 86**](legal-library.html?q=Section+86) — Central Consumer Protection Authority (CCPA)\n\n"
            "💡 **Tip**: Section 35 is the most commonly cited section when filing at the District level. "
            "Always mention the specific deficiency under Section 2(11) in your complaint."
        )

    else:
        return (
            "I'm **Nyaya**, a specialist in **Indian consumer law**. \u2696\ufe0f\n\n"
            "I'm not able to help with general knowledge questions, but I can assist with:\n\n"
            "\u2705 Filing consumer complaints (COPRA 2019)\n"
            "\u2705 E-commerce disputes (Amazon, Flipkart, etc.)\n"
            "\u2705 Insurance claim rejections\n"
            "\u2705 Banking fraud / UPI scams\n"
            "\u2705 Builder delay / real estate issues\n"
            "\u2705 Filing fees, documents, and court procedures\n\n"
            "**Describe your consumer grievance** and I'll guide you on the best legal approach!"
        )


async def _get_ai_response(user_message: str, chat_history: list) -> str:
    """
    Get AI response with priority: OpenAI → Groq → rule-based fallback.
    """
    import asyncio

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append({
            "role": "user" if msg["sender"] == "user" else "assistant",
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": user_message})

    # ── 1. Anthropic (Claude) — highest priority ──────────────────────
    if settings.ANTHROPIC_API_KEY:
        try:
            from anthropic import AsyncAnthropic
            from ..services.mcp_tools import ANTHROPIC_TOOLS, execute_tool
            
            client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            
            # Anthropic handles the system prompt separately, not in the messages array
            anthropic_messages = []
            for msg in chat_history[-6:]:
                anthropic_messages.append({
                    "role": "user" if msg["sender"] == "user" else "assistant",
                    "content": msg["content"]
                })
            anthropic_messages.append({"role": "user", "content": user_message})
            
            # Loop for tool execution
            while True:
                completion = await client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    system=SYSTEM_PROMPT,
                    messages=anthropic_messages,
                    tools=ANTHROPIC_TOOLS,
                    temperature=0.3,
                    max_tokens=800,
                )
                
                if completion.stop_reason == "tool_use":
                    # Assistant wants to use a tool. Append its entire response to messages.
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": completion.content
                    })
                    
                    tool_results = []
                    for block in completion.content:
                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_args = block.input
                            logger.info(f"Claude requested tool: {tool_name}")
                            
                            # Execute the tool
                            result = execute_tool(tool_name, tool_args)
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                            
                    # Return the tool results to Claude
                    anthropic_messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                else:
                    logger.info("Anthropic (Claude) response received")
                    return completion.content[0].text
                    
        except Exception as e:
            logger.warning(f"Anthropic error [{type(e).__name__}]: {e}")

    # ── 2. OpenAI (gpt-4o-mini) ───────────────────────────────────────
    if settings.OPENAI_API_KEY:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            completion = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            logger.info("OpenAI response received")
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI error [{type(e).__name__}]: {e}")

    # ── 3. Groq AsyncGroq (groq >= 0.11) ────────────────────────────
    if settings.GROQ_API_KEY:
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            completion = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            logger.info("Groq AsyncGroq response received")
            return completion.choices[0].message.content
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"AsyncGroq error [{type(e).__name__}]: {e}")

        # ── 4. Groq sync via executor (groq < 0.11) ──────────────────
        try:
            from groq import Groq

            def _sync_groq():
                c = Groq(api_key=settings.GROQ_API_KEY)
                return c.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=800,
                )

            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(None, _sync_groq)
            logger.info("Groq sync response received")
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq sync error [{type(e).__name__}]: {e}")

    # ── 5. Rule-based fallback ───────────────────────────────────────
    logger.info("Using rule-based fallback response")
    return _rule_based_response(user_message)


# Keep old name as alias for backward compat
_get_groq_response = _get_ai_response


# ─── Request / Response Schemas ───────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    """Request schema for chat message."""
    content: str
    case_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Response schema for chat message."""
    message_id: str
    sender: str
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    """Combined user message + assistant response."""
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/message",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send chat message and get legal assistant response"
)
async def send_message(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """
    Send a chat message and receive a legal assistant response.

    Uses Groq Llama 3 if API key is configured, otherwise falls back
    to a comprehensive rule-based Indian consumer law Q&A system.
    """

    try:
        logger.info(f"Chat message from user: {current_user.uid}")

        if not request.content or len(request.content.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message content cannot be empty"
            )

        # Fetch recent chat history for context
        history_query = (
            select(ChatMessage)
            .where(ChatMessage.user_uid == current_user.uid)
            .order_by(ChatMessage.created_at.desc())
            .limit(10)
        )
        history_result = await db.execute(history_query)
        history_msgs = history_result.scalars().all()
        chat_history = [
            {"sender": m.sender, "content": m.content}
            for m in reversed(history_msgs)
        ]

        # Save user message
        user_msg_id = f"MSG-{uuid.uuid4().hex[:10].upper()}"
        user_message = ChatMessage(
            message_id=user_msg_id,
            user_uid=current_user.uid,
            case_id=request.case_id,
            sender="user",
            content=request.content
        )
        db.add(user_message)
        await db.flush()

        # Generate assistant response
        # Check if ANY AI API key is available (Anthropic, OpenAI, or Groq)
        has_valid_groq = settings.GROQ_API_KEY and settings.GROQ_API_KEY not in ["", "your_groq_api_key_here", "your-groq-api-key-here"]
        has_anthropic = bool(settings.ANTHROPIC_API_KEY)
        has_openai = bool(settings.OPENAI_API_KEY)
        
        if has_anthropic or has_openai or has_valid_groq:
            logger.info("Using AI for response")
            response_text = await _get_ai_response(request.content, chat_history)
        else:
            logger.info("Using rule-based fallback (no AI keys configured)")
            response_text = _rule_based_response(request.content)

        # Save assistant message
        assistant_msg_id = f"MSG-{uuid.uuid4().hex[:10].upper()}"
        assistant_message = ChatMessage(
            message_id=assistant_msg_id,
            user_uid=current_user.uid,
            case_id=request.case_id,
            sender="assistant",
            content=response_text
        )
        db.add(assistant_message)
        await db.commit()
        await db.refresh(user_message)
        await db.refresh(assistant_message)

        logger.info(f"Chat exchange saved: {user_msg_id} / {assistant_msg_id}")

        return ChatResponse(
            user_message=ChatMessageResponse(
                message_id=user_msg_id,
                sender="user",
                content=request.content,
                created_at=user_message.created_at
            ),
            assistant_message=ChatMessageResponse(
                message_id=assistant_msg_id,
                sender="assistant",
                content=response_text,
                created_at=assistant_message.created_at
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message"
        )


@router.get(
    "/history",
    response_model=List[ChatMessageResponse],
    summary="Get chat history"
)
async def get_chat_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    case_id: str = None,
    limit: int = 50
) -> List[ChatMessageResponse]:
    """
    Get chat message history for the current user.
    """

    try:
        logger.info(f"Fetching chat history for user: {current_user.uid}")

        query = select(ChatMessage).where(ChatMessage.user_uid == current_user.uid)

        if case_id:
            query = query.where(ChatMessage.case_id == case_id)

        result = await db.execute(
            query.order_by(ChatMessage.created_at.desc()).limit(limit)
        )

        messages = result.scalars().all()

        return [
            ChatMessageResponse(
                message_id=msg.message_id,
                sender=msg.sender,
                content=msg.content,
                created_at=msg.created_at
            )
            for msg in reversed(messages)
        ]

    except Exception as e:
        logger.error(f"Failed to fetch chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chat history"
        )
