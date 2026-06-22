# JUDGELYTICS - FastAPI Backend: Legal Library Router
# Purpose: IPC/BNS/COPRA legal sections and landmark judgements API
# Authors: Rakhi Tiwari (EN23CS302834), Raghav Dubey (EN23CS301816)
# Date: January 2026

"""
Legal library endpoints for Judgelytics backend.

Provides searchable IPC/BNS/COPRA sections and landmark consumer court judgements.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Legal Library"])

# ─── Load static data ─────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"

def _load_sections() -> list:
    try:
        with open(_DATA_DIR / "legal_sections.json", encoding="utf-8") as f:
            return json.load(f)["sections"]
    except Exception as e:
        logger.error(f"Failed to load legal sections: {e}")
        return []

def _load_judgements() -> list:
    try:
        with open(_DATA_DIR / "judgements.json", encoding="utf-8") as f:
            return json.load(f)["judgements"]
    except Exception as e:
        logger.error(f"Failed to load judgements: {e}")
        return []

# Load once at import time
_SECTIONS = _load_sections()
_JUDGEMENTS = _load_judgements()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class LegalSection(BaseModel):
    id: str
    act: str
    number: str
    title: str
    description: str
    keywords: List[str]
    category: str


class Judgement(BaseModel):
    id: str
    title: str
    court: str
    year: int
    citation: str
    sector: str
    outcome: str
    summary: str
    key_principle: str
    sections: List[str]
    tags: List[str]


class SectionsList(BaseModel):
    total: int
    sections: List[LegalSection]


class JudgementsList(BaseModel):
    total: int
    judgements: List[Judgement]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get(
    "/sections",
    response_model=SectionsList,
    summary="Get all legal sections"
)
async def get_sections(
    query: Optional[str] = Query(None, description="Search keyword"),
    act: Optional[str] = Query(None, description="Filter by Act (e.g. 'COPRA 2019', 'BNS 2023')"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, le=100)
) -> SectionsList:
    """
    Get searchable list of IPC/BNS/COPRA legal sections relevant to consumer cases.
    """
    sections = _SECTIONS.copy()

    # Filter by act
    if act:
        sections = [s for s in sections if act.lower() in s["act"].lower()]

    # Filter by category
    if category:
        sections = [s for s in sections if category.lower() in s["category"].lower()]

    # Full-text search
    if query:
        q = query.lower()
        sections = [
            s for s in sections
            if (
                q in s["title"].lower()
                or q in s["description"].lower()
                or q in s["number"].lower()
                or any(q in kw.lower() for kw in s["keywords"])
            )
        ]

    sections = sections[:limit]

    return SectionsList(total=len(sections), sections=[LegalSection(**s) for s in sections])


@router.get(
    "/sections/{section_id}",
    response_model=LegalSection,
    summary="Get a specific legal section by ID"
)
async def get_section(section_id: str) -> LegalSection:
    """
    Get detailed information about a specific legal section.
    """
    for s in _SECTIONS:
        if s["id"] == section_id:
            return LegalSection(**s)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Section '{section_id}' not found"
    )


@router.get(
    "/judgements",
    response_model=JudgementsList,
    summary="Get landmark consumer court judgements"
)
async def get_judgements(
    query: Optional[str] = Query(None, description="Search keyword"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    outcome: Optional[str] = Query(None, description="Filter by outcome (Allowed/Dismissed)"),
    limit: int = Query(20, le=50)
) -> JudgementsList:
    """
    Get searchable landmark Indian consumer court judgements.
    """
    judgements = _JUDGEMENTS.copy()

    # Filter by sector
    if sector:
        judgements = [j for j in judgements if sector.lower() in j["sector"].lower()]

    # Filter by outcome
    if outcome:
        judgements = [j for j in judgements if j["outcome"].lower() == outcome.lower()]

    # Full-text search
    if query:
        q = query.lower()
        judgements = [
            j for j in judgements
            if (
                q in j["title"].lower()
                or q in j["summary"].lower()
                or q in j["key_principle"].lower()
                or any(q in tag.lower() for tag in j["tags"])
                or q in j["sector"].lower()
            )
        ]

    judgements = judgements[:limit]

    return JudgementsList(
        total=len(judgements),
        judgements=[Judgement(**j) for j in judgements]
    )


@router.get(
    "/judgements/{judgement_id}",
    response_model=Judgement,
    summary="Get a specific judgement"
)
async def get_judgement(judgement_id: str) -> Judgement:
    """
    Get full details of a specific landmark judgement.
    """
    for j in _JUDGEMENTS:
        if j["id"] == judgement_id:
            return Judgement(**j)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Judgement '{judgement_id}' not found"
    )


@router.get(
    "/categories",
    summary="Get list of available section categories"
)
async def get_categories():
    """Get unique categories for filtering."""
    categories = sorted(set(s["category"] for s in _SECTIONS))
    acts = sorted(set(s["act"] for s in _SECTIONS))
    sectors = sorted(set(j["sector"] for j in _JUDGEMENTS))

    return {
        "categories": categories,
        "acts": acts,
        "sectors": sectors
    }
