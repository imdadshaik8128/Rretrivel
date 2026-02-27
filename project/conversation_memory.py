"""
conversation_memory.py — Session Conversation Memory Layer
============================================================
Stores the last N query/answer turns per session and provides
three services to the rest of the pipeline:

  1. enrich_query(raw_query) → str
        Resolves vague pronouns in CONCEPT follow-ups into a
        self-contained query string.
        Only touches free-text. Never appends chapter/activity numbers
        to text — those are handled by restore_structured_filters().

  2. restore_structured_filters(parsed_dict, raw_query) → (dict, bool)
        After sanitize() runs, hard-restores chapter_number and
        activity_number from memory IF the query is a follow-up AND
        the user didn't explicitly mention a different number.
        This bypasses the SLM re-parsing problem entirely.

  3. build_context_block() → str
        Returns a compact plain-text block of the last N turns
        for injection into the LLM generator prompt.

────────────────────────────────────────────────────────────────
WHY restore_structured_filters EXISTS (the core bug fix)
────────────────────────────────────────────────────────────────
The original design appended "chapter 1" as TEXT to the enriched query,
then relied on the SLM to re-parse it as chapter_number (int).
This failed because:
  - SLM sometimes parses "chapter 1" → chapter_name (str), not chapter_number (int)
  - sanitize() drops chapter_name since only chapter_number is used downstream
  - The chapter filter disappears → retriever searches ALL chapters → wrong results

The correct fix:
  - enrich_query()               → enriches free-text pronouns ONLY
  - restore_structured_filters() → directly writes chapter_number / activity_number
                                   into parsed_dict AFTER sanitize, as hard integers

────────────────────────────────────────────────────────────────
FOLLOW-UP DETECTION
────────────────────────────────────────────────────────────────
A query is treated as a follow-up if it contains ANY of:
  Pronouns     : it, its, this, that, these, those
  Demonstratives: "the activity", "the topic", "the concept", "the same"
  Continuation : still, again, further, more clearly, more detail,
                 can you explain, tell me more, elaborate, what about

Design constraints respected:
  - Structure beats similarity: chapter/activity restored as hard integers, never re-parsed.
  - SLM never performs retrieval: enrichment runs before SLM parser.
  - Subject leakage is a critical bug: subject always injected from session in Main.py.
  - User's explicit numbers always win: if user types "activity 3", memory doesn't override.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

MAX_TURNS = 5

# ── Follow-up signal detection ─────────────────────────────────────────────────
_FOLLOWUP_TRIGGERS = re.compile(
    r"\b("
    r"it|its|this|that|these|those"
    r"|the\s+activity|the\s+exercise|the\s+topic|the\s+concept|the\s+same"
    r"|still|again|further"
    r"|more\s+clearly|more\s+detail|more\s+about|explain\s+more|elaborate"
    r"|can\s+you\s+explain|tell\s+me\s+more"
    r"|what\s+about|how\s+about"
    r")\b",
    re.IGNORECASE,
)

# Detect explicit chapter/activity numbers in the raw query
_EXPLICIT_CHAPTER  = re.compile(r"\b(chapter|ch\.?)\s*(\d+)\b", re.IGNORECASE)
_EXPLICIT_ACTIVITY = re.compile(r"\bactivity\s*(\d+)\b",         re.IGNORECASE)
_EXPLICIT_EXERCISE = re.compile(r"\bexercise\s*([\d.]+)\b",      re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationTurn:
    """
    One completed turn stored in memory.
    chapter_number and activity_number come from the RETRIEVER's chunks —
    the authoritative values actually used for retrieval, not the SLM's output.
    """
    raw_query:       str
    enriched_query:  str
    topic_keywords:  list[str]
    chapter_number:  Optional[str]   # from retriever (authoritative)
    chapter_title:   Optional[str]
    activity_number: Optional[str]   # from retriever (authoritative)
    subject:         str
    answer_summary:  str
    answer_type:     str


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

_STOP = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","up",
    "about","into","through","during","before","after","above","below",
    "between","each","and","or","but","if","while","although","because",
    "since","what","how","why","when","where","which","who","that","this",
    "these","those","it","its","i","me","my","we","our","you","your","he",
    "she","they","them","their","chapter","activity","exercise","explain",
    "tell","describe","give","show","find","get","let","make","go","also",
    "then","just","very","more","some","any","all","no","not","so","such",
    "both","either","neither","nor","yet","though","still","again",
}


def _extract_topic_keywords(
    query:           str,
    spoken_answer:   str,
    parsed_topic:    str = "",
    activity_number: Optional[str] = None,
    chapter_number:  Optional[str] = None,
) -> list[str]:
    """
    Extract topic keywords for display/logging and concept pronoun resolution.
    For reference queries, stores a structured label like "activity 1 chapter 1".
    """
    keywords: list[str] = []

    # Reference queries: build a structured label as primary keyword
    if activity_number and str(activity_number).strip() not in ("", "None"):
        label = f"activity {activity_number}"
        if chapter_number and str(chapter_number).strip() not in ("", "None"):
            label += f" chapter {chapter_number}"
        keywords.append(label)

    # Clean SLM topic (strip "Activity N:" prefixes)
    if parsed_topic and len(parsed_topic.strip()) > 2:
        clean = re.sub(r"^activity\s*\d+[:\s]*", "", parsed_topic, flags=re.IGNORECASE).strip()
        if clean and clean.lower() not in keywords:
            keywords.append(clean.lower())
        for tok in re.findall(r"[a-zA-Z]{4,}", parsed_topic.lower()):
            if tok not in _STOP and tok not in keywords:
                keywords.append(tok)

    # Significant words from query
    for tok in re.findall(r"[a-zA-Z]{4,}", query.lower()):
        if tok not in _STOP and tok not in keywords:
            keywords.append(tok)

    # First sentence of spoken answer
    first = re.split(r"[.!?]", spoken_answer)[0] if spoken_answer else ""
    for tok in re.findall(r"[a-zA-Z]{4,}", first.lower()):
        if tok not in _STOP and tok not in keywords:
            keywords.append(tok)

    return keywords[:8]


def _summarise_answer(spoken_answer: str, max_sentences: int = 3) -> str:
    if not spoken_answer:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", spoken_answer.strip())
    return " ".join(sentences[:max_sentences]).strip()


def _is_followup(query: str) -> bool:
    """True if the query contains any follow-up trigger signal."""
    return bool(_FOLLOWUP_TRIGGERS.search(query))


# ══════════════════════════════════════════════════════════════════════════════
# Main memory class
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Session-scoped conversation memory. Instantiate once in Main.py.

    Call order in run_pipeline():
      1. enriched = memory.enrich_query(raw_query)          # before SLM parse
      2. ... SLM parse → sanitize → inject subject ...
      3. parsed_dict, used = memory.restore_structured_filters(parsed_dict, raw_query)
      4. ... retrieve → generate → display ...
      5. memory.add_turn(...)                                # after successful generate
    """

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self._max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        log.info("ConversationMemory initialised (max_turns=%d)", max_turns)

    # ──────────────────────────────────────────────────────────────────────────
    # add_turn
    # ──────────────────────────────────────────────────────────────────────────

    def add_turn(
        self,
        raw_query:      str,
        enriched_query: str,
        parsed_topic:   str,
        chunks:         list,
        answer,
        subject:        str,
    ) -> None:
        """Store a completed turn after successful generation."""
        chapter_number  = None
        chapter_title   = None
        activity_number = None

        if chunks:
            chapter_number = str(chunks[0].chapter_number) if chunks[0].chapter_number else None
            chapter_title  = chunks[0].chapter_title or None
            for c in chunks:
                act = str(c.activity_number).strip()
                if act and act not in ("", "None", "none"):
                    activity_number = act
                    break

        keywords = _extract_topic_keywords(
            query           = raw_query,
            spoken_answer   = answer.spoken_answer,
            parsed_topic    = parsed_topic,
            activity_number = activity_number,
            chapter_number  = chapter_number,
        )
        summary = _summarise_answer(answer.spoken_answer, max_sentences=3)

        turn = ConversationTurn(
            raw_query       = raw_query,
            enriched_query  = enriched_query,
            topic_keywords  = keywords,
            chapter_number  = chapter_number,
            chapter_title   = chapter_title,
            activity_number = activity_number,
            subject         = subject,
            answer_summary  = summary,
            answer_type     = answer.answer_type,
        )
        self._turns.append(turn)
        log.info(
            "Memory: stored turn %d | activity=%s | chapter=%s | keywords=%s",
            len(self._turns), activity_number, chapter_number, keywords[:2],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # enrich_query  (free-text pronoun resolution for CONCEPT queries only)
    # ──────────────────────────────────────────────────────────────────────────

    def enrich_query(self, raw_query: str) -> str:
        """
        Resolve topic-level pronouns for concept follow-ups.

        Does NOT handle reference (activity/exercise) follow-ups — those are
        handled entirely in restore_structured_filters() because what matters
        there is the activity/chapter NUMBER, not a text label.

        Does NOT append chapter numbers to the text string (that was the old
        bug — SLM would mis-parse them as chapter_name instead of chapter_number).
        """
        if not self._turns:
            return raw_query

        last = self._turns[-1]

        # Reference query follow-ups: no text enrichment needed
        # restore_structured_filters() will handle the numbers
        if last.answer_type == "reference":
            return raw_query

        # For concept queries, find the primary topic keyword (skip activity labels)
        primary_topic = ""
        for kw in last.topic_keywords:
            if not re.match(r"^activity\s+\d+", kw, re.IGNORECASE):
                primary_topic = kw
                break

        if not primary_topic or not _is_followup(raw_query):
            return raw_query

        enriched = raw_query.strip()
        changed  = False

        pronoun_patterns = [
            (r"\bit\b",   primary_topic),
            (r"\bthis\b(?!\s+(chapter|activity|exercise|section))", primary_topic),
            (r"\bthat\b(?!\s+(chapter|activity|exercise|section))", primary_topic),
            (r"\bthe\s+(topic|concept|subject|process|phenomenon)\b", primary_topic),
        ]
        for pattern, replacement in pronoun_patterns:
            new = re.sub(pattern, replacement, enriched, flags=re.IGNORECASE)
            if new != enriched:
                enriched = new
                changed  = True
                log.info("Memory: text enriched '%s' → '%s'", raw_query, enriched)
                break

        if not changed and len(enriched.split()) <= 4:
            if primary_topic.lower() not in enriched.lower():
                enriched = f"{enriched} of {primary_topic}"
                log.info("Memory: short query enriched → '%s'", enriched)

        return enriched

    # ──────────────────────────────────────────────────────────────────────────
    # restore_structured_filters  ← THE CORE FIX
    # ──────────────────────────────────────────────────────────────────────────

    def restore_structured_filters(
        self,
        parsed_dict: dict,
        raw_query:   str,
    ) -> tuple[dict, bool]:
        """
        Restore chapter_number and/or activity_number from memory into the
        parsed_dict AFTER sanitize() has already run.

        Called in Main.py between sanitize() and subject injection.

        Restores a field if ALL conditions are met:
          1. Query is a follow-up (trigger signal present)
          2. Memory has that field from the last turn
          3. The user did NOT explicitly type a different number in this query
          4. That field is currently None in parsed_dict (sanitizer zeroed it)

        Returns (updated_dict, memory_was_used).
        """
        if not self._turns:
            return parsed_dict, False

        if not _is_followup(raw_query):
            return parsed_dict, False

        last   = self._turns[-1]
        result = dict(parsed_dict)
        used   = False

        # ── Restore chapter_number ─────────────────────────────────────────────
        if (
            last.chapter_number is not None
            and result.get("chapter_number") is None
        ):
            explicit = _EXPLICIT_CHAPTER.search(raw_query)
            if not explicit:
                try:
                    result["chapter_number"] = int(last.chapter_number)
                    used = True
                    log.info(
                        "Memory: restored chapter_number=%s into parsed_dict",
                        last.chapter_number,
                    )
                except (ValueError, TypeError):
                    pass

        # ── Restore activity_number ────────────────────────────────────────────
        if (
            last.activity_number is not None
            and result.get("activity_number") is None
        ):
            explicit = _EXPLICIT_ACTIVITY.search(raw_query)
            if not explicit:
                try:
                    result["activity_number"] = int(last.activity_number)
                    if result.get("chunk_type") in (None, "", "unknown", "theory"):
                        result["chunk_type"] = "activity"
                    used = True
                    log.info(
                        "Memory: restored activity_number=%s into parsed_dict",
                        last.activity_number,
                    )
                except (ValueError, TypeError):
                    pass

        return result, used

    # ──────────────────────────────────────────────────────────────────────────
    # build_context_block
    # ──────────────────────────────────────────────────────────────────────────

    def build_context_block(self) -> str:
        """
        Compact plain-text block of last N turns for LLM prompt injection.
        Returns empty string if no turns stored.
        """
        if not self._turns:
            return ""

        lines = ["--- CONVERSATION HISTORY (context only — not textbook content) ---"]
        for i, turn in enumerate(self._turns, 1):
            ref = ""
            if turn.activity_number:
                ref = f" (Activity {turn.activity_number}, Chapter {turn.chapter_number})"
            lines.append(
                f"[Turn {i}] Student asked: {turn.raw_query}{ref}\n"
                f"Answer: {turn.answer_summary}"
            )
            lines.append("")
        lines.append("--- END OF HISTORY ---")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # clear / properties
    # ──────────────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._turns.clear()
        log.info("ConversationMemory cleared.")

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self._turns[-1] if self._turns else None

    def __repr__(self) -> str:
        lt = self.last_turn
        return (
            f"ConversationMemory(turns={self.turn_count}/{self._max_turns}, "
            f"last_activity={lt.activity_number if lt else None}, "
            f"last_chapter={lt.chapter_number if lt else None})"
        )