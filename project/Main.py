"""
main.py — Textbook Q&A Chat Interface
=======================================
Run:
    python main.py

Session flow:
  1. System loads (retriever + generator + TTS)
  2. User selects a subject from a numbered menu
  3. Subject is locked for the session — all queries filtered to that subject
  4. Each query runs: parse → retrieve → generate → display (markdown) + TTS (spoken)
  5. Commands: 'switch' to change subject | 'help' | 'exit'

Pipeline per query:
    query_parser_v2  →  Retriever  →  Generator  →  display_answer + speak()
"""

from __future__ import annotations

import json
import sys
import time

# ── Rich terminal renderer ─────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ── TTS ────────────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

from query_parser_v2 import parse_query_with_slm
from retriever import Retriever, AmbiguityError
from parse_sanitizer import sanitize  
from generator import Generator

# ── Constants ──────────────────────────────────────────────────────────────────
TTS_RATE   = 165
TTS_VOLUME = 0.9

# Must match subject field values in all_chunks.json exactly
AVAILABLE_SUBJECTS = [
    "Biology",
    "Economics",
    "Geography",
    "History",
    "Maths_sem_1",
    "Maths_sem_2",
    "Physics",
    "Social_political",
]


# ══════════════════════════════════════════════════════════════════════════════
# ANSI colour helpers (fallback when rich is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

BOLD   = lambda t: _c("1",  t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
RED    = lambda t: _c("31", t)
DIM    = lambda t: _c("2",  t)


# ══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print(text: str, style: str = "") -> None:
    if RICH_AVAILABLE:
        console.print(text, style=style)
    else:
        print(text)

def _rule(title: str = "") -> None:
    if RICH_AVAILABLE:
        console.print(Rule(title, style="bold cyan"))
    else:
        label = f"  {title}  " if title else ""
        print(f"\n{'─' * 64}{label}")


# ══════════════════════════════════════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          Textbook Q&A  —  RAG System  (offline)              ║
║          Retrieval · Generation · Display · TTS              ║
╚══════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console.print(banner, style="bold cyan")
    else:
        print(BOLD(banner))


# ══════════════════════════════════════════════════════════════════════════════
# Subject selection
# ══════════════════════════════════════════════════════════════════════════════

def select_subject() -> str:
    """Numbered menu — returns the chosen subject string."""
    if RICH_AVAILABLE:
        console.print("\n  [bold]Select a subject to study:[/bold]\n")
        for i, subj in enumerate(AVAILABLE_SUBJECTS, 1):
            console.print(f"    [cyan]{i}[/cyan].  {subj}")
        console.print()
    else:
        print(BOLD("\n  Select a subject to study:\n"))
        for i, subj in enumerate(AVAILABLE_SUBJECTS, 1):
            print(f"    {CYAN(str(i))}.  {subj}")
        print()

    while True:
        try:
            choice = input(BOLD("  Enter number › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_SUBJECTS):
                selected = AVAILABLE_SUBJECTS[idx]
                if RICH_AVAILABLE:
                    console.print(f"\n  [bold green]✓ Subject locked: {selected}[/bold green]\n")
                else:
                    print(GREEN(f"\n  ✓ Subject locked: {selected}\n"))
                return selected

        if RICH_AVAILABLE:
            console.print(f"  [red]Invalid. Enter a number 1–{len(AVAILABLE_SUBJECTS)}.[/red]")
        else:
            print(RED(f"  Invalid. Enter a number 1–{len(AVAILABLE_SUBJECTS)}."))


# ══════════════════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════════════════

def _init_tts():
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate",   TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        voices = engine.getProperty("voices")
        if voices:
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
        return engine
    except Exception as e:
        _print(f"  ⚠  TTS init failed: {e}", style="yellow")
        return None

def speak(engine, text: str) -> None:
    if engine is None:
        _print("  (TTS unavailable — install pyttsx3 + espeak)", style="dim yellow")
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        _print(f"  (TTS error: {e})", style="dim yellow")


# ══════════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════════

def display_answer(answer) -> None:
    """Render the full GeneratedAnswer — markdown display then spoken panel."""

    _rule()

    # Answer type badge
    if RICH_AVAILABLE:
        badge = (
            "[bold green]● REFERENCE LOOKUP[/bold green]"
            if answer.answer_type == "reference"
            else "[bold blue]● CONCEPT EXPLANATION[/bold blue]"
        )
        console.print(badge)
    else:
        print(f"[ {answer.answer_type.upper()} ]")

    # Low confidence warning
    if answer.low_confidence_warning:
        if RICH_AVAILABLE:
            console.print(Panel(
                f"⚠  {answer.low_confidence_warning}",
                style="bold yellow",
                title="Low Confidence Warning",
            ))
        else:
            print(YELLOW(f"\n⚠  {answer.low_confidence_warning}"))

    # Confidence bar
    pct = int(answer.confidence * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    _print(f"  Confidence: [{bar}] {pct}%", style="dim")

    # ── Display answer (markdown) ──────────────────────────────────────────────
    _rule("DISPLAY ANSWER")
    if RICH_AVAILABLE:
        console.print(Markdown(answer.display_answer_markdown))
    else:
        import re
        plain = re.sub(r"#{1,6}\s+", "", answer.display_answer_markdown)
        plain = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", plain)
        plain = re.sub(r"`(.+?)`", r"\1", plain)
        print(plain)

    # ── Citations ──────────────────────────────────────────────────────────────
    if answer.citations:
        _rule("SOURCES")
        for i, c in enumerate(answer.citations, 1):
            act = (
                f"  (Activity {c.activity_number})"
                if c.activity_number.strip() not in ("", "None")
                else ""
            )
            line = (
                f"  [{i}] Chapter {c.chapter_number}"
                + (f" — {c.section_title}" if c.section_title else "")
                + act
                + f"  [{c.chunk_type}]"
            )
            _print(line, style="dim cyan")

    _print(f"\n  Filter path: {answer.filter_path}", style="dim")

    # ── Spoken answer panel ────────────────────────────────────────────────────
    _rule("SPOKEN ANSWER  (TTS)")
    if RICH_AVAILABLE:
        console.print(Panel(
            answer.spoken_answer,
            style="italic green",
            title="🔊 Speaking …",
        ))
    else:
        print(f"\n🔊  {answer.spoken_answer}\n")

    _rule()


def print_help(subject: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"""
  [bold]Commands:[/bold]
    [cyan]switch[/cyan]   — change subject  (currently: [green]{subject}[/green])
    [cyan]help[/cyan]     — show this message
    [cyan]exit[/cyan]     — quit

  [bold]Example queries:[/bold]
    explain activity 2 from chapter 1
    what is photosynthesis chapter 3
    solve exercise 3.1 chapter 4
    how does osmosis work chapter 3
""")
    else:
        print(f"""
  Commands:
    switch   — change subject  (currently: {subject})
    help     — show this message
    exit     — quit

  Example queries:
    explain activity 2 from chapter 1
    what is photosynthesis chapter 3
    solve exercise 3.1 chapter 4
""")


# ══════════════════════════════════════════════════════════════════════════════
# Per-query pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    query:          str,
    active_subject: str,
    retriever:      Retriever,
    generator:      Generator,
    tts_engine,
) -> None:
    """parse → inject subject → retrieve → generate → display → speak"""

    # ── Step 1: Parse ──────────────────────────────────────────────────────────
    try:
        raw_parse   = parse_query_with_slm(query)
        parsed_dict = json.loads(raw_parse)
    except Exception as e:
        _print(f"  ✗ Parser error: {e}", style="bold red")
        return

    # Strip hallucinated fields the user never mentioned  ← ADD THIS
    parsed_dict = sanitize(parsed_dict, query)

    # Inject session subject — hard override
    parsed_dict["subject"] = active_subject
    _print(f"  ↳ Parsed: {json.dumps(parsed_dict)}", style="dim")
    print()

    # ── Step 2: Retrieve ───────────────────────────────────────────────────────
    _print("  Retrieving …", style="dim")
    t0 = time.perf_counter()
    chunks, ret_err = retriever.retrieve_safe(parsed_dict, query)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    if ret_err:
        if RICH_AVAILABLE:
            console.print(Panel(
                f"✗  {ret_err}\n\nCheck chapter number or activity number.",
                style="bold red",
                title="Retrieval Error",
            ))
        else:
            print(RED(f"\n  ✗ {ret_err}"))
        return

    _print(f"  ✓ Retrieved {len(chunks)} chunks in {retrieval_ms}ms", style="dim green")

    # ── Step 3: Generate ───────────────────────────────────────────────────────
    _print("  Generating answer via Ollama …", style="dim")
    t1 = time.perf_counter()
    answer, gen_err = generator.generate_safe(chunks, query)
    generation_ms = int((time.perf_counter() - t1) * 1000)

    if gen_err:
        if RICH_AVAILABLE:
            console.print(Panel(f"✗  {gen_err}", style="bold red", title="Generation Error"))
        else:
            print(RED(f"\n  ✗ {gen_err}"))
        if "not running" in gen_err.lower() or "not found" in gen_err.lower():
            _print("  → Run:  ollama serve", style="yellow")
            _print("  → Pull: ollama pull qwen2.5:0.5b-instruct", style="yellow")
        return

    _print(
        f"  ✓ Generated in {generation_ms}ms  |  "
        f"type={answer.answer_type}  |  confidence={answer.confidence:.0%}",
        style="dim green",
    )

    # ── Step 4: Display (markdown) ─────────────────────────────────────────────
    # ── Step 5: Speak  (TTS)       ─────────────────────────────────────────────
    display_answer(answer)
    speak(tts_engine, answer.spoken_answer)


# ══════════════════════════════════════════════════════════════════════════════
# Main chat loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print_banner()

    # ── Load everything once ───────────────────────────────────────────────────
    _print("⏳  Loading retriever (bi-encoder + cross-encoder) …\n", style="yellow")
    retriever = Retriever()

    generator  = Generator()

    _print("🔊  Initialising TTS (pyttsx3) …", style="dim")
    tts_engine = _init_tts()

    if not TTS_AVAILABLE:
        _print(
            "  ⚠  pyttsx3 not found — spoken answer will display but not play.\n"
            "     Install: pip install pyttsx3\n"
            "     Linux  : sudo apt-get install espeak",
            style="yellow",
        )

    _print("\n✓  System ready.\n", style="bold green")

    # ── Subject selection (runs once, re-runs on 'switch') ─────────────────────
    active_subject = select_subject()

    if RICH_AVAILABLE:
        console.print(
            f"  Type [cyan]help[/cyan] for commands. "
            f"Type [cyan]switch[/cyan] to change subject.\n"
        )
    else:
        print("  Type 'help' for commands. Type 'switch' to change subject.\n")

    _rule()

    # ── Chat loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            if RICH_AVAILABLE:
                console.print(
                    f"\n[bold cyan][{active_subject}] You ›[/bold cyan] ", end=""
                )
                query = input()
            else:
                query = input(BOLD(f"\n[{active_subject}] You › "))

            query = query.strip()

        except (EOFError, KeyboardInterrupt):
            print()
            _print("\n  Goodbye!\n", style="bold cyan")
            sys.exit(0)

        if not query:
            continue

        cmd = query.lower()

        if cmd in {"exit", "quit", "q"}:
            _print("\n  Goodbye!\n", style="bold cyan")
            sys.exit(0)

        if cmd == "switch":
            print()
            active_subject = select_subject()
            _rule()
            continue

        if cmd == "help":
            print_help(active_subject)
            continue

        print()
        run_pipeline(query, active_subject, retriever, generator, tts_engine)


if __name__ == "__main__":
    main()