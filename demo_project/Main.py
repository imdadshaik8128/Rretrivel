"""
main.py — End-to-end CLI Runner
=================================
Pipeline:
    user query
        │
        ▼
    query_parser_v2   (SLM — extracts metadata)
        │
        ▼
    Retriever         (metadata filter → bi-encoder → cross-encoder)
        │
        ▼
    Generator         (Ollama local LLM → dual format answer)
        │
        ├──► Terminal display  (markdown rendered in terminal)
        └──► pyttsx3 TTS       (speaks the spoken_answer)

Install dependencies:
    pip install pyttsx3 rich requests sentence-transformers
    ollama pull qwen2.5:0.5b-instruct
"""

from __future__ import annotations

import json
import sys
import time

# ── Terminal markdown renderer ─────────────────────────────────────────────────
# 'rich' renders markdown beautifully in the terminal.
# Install: pip install rich
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── TTS engine ─────────────────────────────────────────────────────────────────
# pyttsx3 works fully offline, no API key, no internet.
# Install: pip install pyttsx3
# On Linux also: sudo apt-get install espeak
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

from retriever import Retriever, AmbiguityError
from generator import Generator

# ── Constants ──────────────────────────────────────────────────────────────────
TTS_RATE      = 165    # words per minute (default ~200, slower = clearer)
TTS_VOLUME    = 0.9    # 0.0 – 1.0
DIVIDER       = "═" * 65


# ══════════════════════════════════════════════════════════════════════════════
# Terminal display helpers
# ══════════════════════════════════════════════════════════════════════════════

console = Console() if RICH_AVAILABLE else None


def _print(text: str, style: str = ""):
    if RICH_AVAILABLE:
        console.print(text, style=style)
    else:
        print(text)


def _rule(title: str = ""):
    if RICH_AVAILABLE:
        console.print(Rule(title, style="bold cyan"))
    else:
        print(f"\n{'─' * 65}  {title}")


def display_answer(answer) -> None:
    """
    Renders the GeneratedAnswer to terminal.
    Uses 'rich' for markdown if available, falls back to plain print.
    """
    _rule()

    # ── Answer type badge ──────────────────────────────────────────────────────
    badge = (
        "[bold green]● REFERENCE LOOKUP[/bold green]"
        if answer.answer_type == "reference"
        else "[bold blue]● CONCEPT EXPLANATION[/bold blue]"
    )
    if RICH_AVAILABLE:
        console.print(badge)
    else:
        print(f"[ {answer.answer_type.upper()} ]")

    # ── Low confidence warning ─────────────────────────────────────────────────
    if answer.low_confidence_warning:
        if RICH_AVAILABLE:
            console.print(
                Panel(
                    f"⚠  {answer.low_confidence_warning}",
                    style="bold yellow",
                    title="Low Confidence Warning",
                )
            )
        else:
            print(f"\n⚠  WARNING: {answer.low_confidence_warning}")

    # ── Confidence bar ─────────────────────────────────────────────────────────
    pct = int(answer.confidence * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    conf_line = f"Confidence: [{bar}] {pct}%"
    _print(conf_line, style="dim")

    # ── Display answer (markdown) ──────────────────────────────────────────────
    _rule("DISPLAY ANSWER")
    if RICH_AVAILABLE:
        console.print(Markdown(answer.display_answer_markdown))
    else:
        # Plain fallback — strip common markdown symbols
        import re
        plain = re.sub(r"#{1,6}\s+", "", answer.display_answer_markdown)
        plain = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", plain)
        plain = re.sub(r"`(.+?)`", r"\1", plain)
        print(plain)

    # ── Citations ──────────────────────────────────────────────────────────────
    if answer.citations:
        _rule("SOURCES")
        for i, c in enumerate(answer.citations, 1):
            line = (
                f"  [{i}] Chapter {c.chapter_number}"
                + (f" — {c.section_title}" if c.section_title else "")
                + (f"  (Activity {c.activity_number})" if c.activity_number.strip() not in ("", "None") else "")
                + f"  [{c.chunk_type}]"
            )
            _print(line, style="dim cyan")

    # ── Filter path (provenance) ───────────────────────────────────────────────
    _print(f"\n  Filter path : {answer.filter_path}", style="dim")
    _rule()


def display_spoken_text(spoken: str) -> None:
    """Show the spoken answer in terminal before/during TTS."""
    _rule("SPOKEN ANSWER  (TTS)")
    if RICH_AVAILABLE:
        console.print(
            Panel(spoken, style="italic green", title="🔊 Speaking …")
        )
    else:
        print(f"\n🔊 {spoken}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TTS engine
# ══════════════════════════════════════════════════════════════════════════════

def _init_tts():
    """Initialise pyttsx3 engine with configured rate and volume."""
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate",   TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)

        # Pick a clear voice if multiple are available
        voices = engine.getProperty("voices")
        if voices:
            # Prefer English voice
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break

        return engine
    except Exception as e:
        print(f"⚠  TTS init failed: {e}")
        return None


def speak(engine, text: str) -> None:
    """Speak text using pyttsx3. Silently skips if engine is unavailable."""
    if engine is None:
        _print("  (TTS unavailable — install pyttsx3 and espeak)", style="dim yellow")
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        _print(f"  (TTS error: {e})", style="dim yellow")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    query:     str,
    retriever: Retriever,
    generator: Generator,
    tts_engine,
) -> None:
    """
    Full pipeline for one query:
      parse → retrieve → generate → display → speak
    """
    print(f"\n{DIVIDER}")
    _print(f"  Query : {query}", style="bold white")
    print(DIVIDER)

    # ── Step 1: Parse query ────────────────────────────────────────────────────
    try:
        from query_parser_v2 import parse_query_with_slm
        raw_parse  = parse_query_with_slm(query)
        parsed_dict = json.loads(raw_parse)
        _print(f"  Parsed : {json.dumps(parsed_dict)}", style="dim")
    except Exception as e:
        _print(f"  ⚠  Query parser error: {e}", style="bold red")
        return

    # ── Step 2: Retrieve chunks ────────────────────────────────────────────────
    _print("\n  Retrieving chunks …", style="dim")
    t0 = time.perf_counter()
    chunks, ret_err = retriever.retrieve_safe(parsed_dict, query)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    if ret_err:
        _print(f"\n  ⚠  Retrieval failed: {ret_err}", style="bold red")
        _print(
            "  → Check subject name, chapter number, or activity number.",
            style="yellow",
        )
        return

    _print(
        f"  Retrieved {len(chunks)} chunks in {retrieval_ms}ms",
        style="dim green",
    )

    # ── Step 3: Generate answer ────────────────────────────────────────────────
    _print("\n  Generating answer via Ollama …", style="dim")
    t1 = time.perf_counter()
    answer, gen_err = generator.generate_safe(chunks, query)
    generation_ms = int((time.perf_counter() - t1) * 1000)

    if gen_err:
        _print(f"\n  ⚠  Generation failed: {gen_err}", style="bold red")
        if "not running" in gen_err:
            _print(
                "  → Start Ollama:  ollama serve",
                style="yellow",
            )
            _print(
                "  → Pull model  :  ollama pull qwen2.5:0.5b-instruct",
                style="yellow",
            )
        return

    _print(
        f"  Generated in {generation_ms}ms  |  "
        f"type={answer.answer_type}  |  "
        f"confidence={answer.confidence:.0%}",
        style="dim green",
    )

    # ── Step 4: Display ────────────────────────────────────────────────────────
    display_answer(answer)

    # ── Step 5: TTS ───────────────────────────────────────────────────────────
    display_spoken_text(answer.spoken_answer)
    speak(tts_engine, answer.spoken_answer)


# ══════════════════════════════════════════════════════════════════════════════
# Interactive REPL loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _print("\n  Initialising system …", style="bold cyan")

    # Load retriever once (embeddings computed at startup)
    _print("  Loading retriever (bi-encoder + cross-encoder) …", style="dim")
    retriever = Retriever()

    # Generator is stateless
    generator = Generator()

    # Init TTS
    _print("  Initialising TTS engine (pyttsx3) …", style="dim")
    tts_engine = _init_tts()

    if not TTS_AVAILABLE:
        _print(
            "  ⚠  pyttsx3 not installed. Spoken answer will be shown but not spoken.\n"
            "     Install: pip install pyttsx3\n"
            "     Linux  : sudo apt-get install espeak",
            style="yellow",
        )

    _print("\n  ✓ System ready.\n", style="bold green")
    _print("  Type your question and press Enter.", style="dim")
    _print("  Commands:  'quit' or 'exit' to stop  |  'json' to see raw output\n", style="dim")

    show_json = False  # toggle with 'json' command

    while True:
        try:
            # Prompt
            if RICH_AVAILABLE:
                console.print("[bold cyan]You >[/bold cyan] ", end="")
                query = input()
            else:
                query = input("You > ")

            query = query.strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                _print("\n  Goodbye!\n", style="bold cyan")
                break

            # Toggle raw JSON output
            if query.lower() == "json":
                show_json = not show_json
                state = "ON" if show_json else "OFF"
                _print(f"  Raw JSON output: {state}", style="dim yellow")
                continue

            # Run the full pipeline
            run_pipeline(query, retriever, generator, tts_engine)

            # Optionally show raw JSON
            if show_json:
                _rule("RAW JSON OUTPUT")
                # Re-run just generator to get the dict (last answer not stored)
                # Better: store last answer in run_pipeline
                _print("  (Run with json mode — last answer above)", style="dim")

        except KeyboardInterrupt:
            _print("\n\n  Interrupted. Type 'quit' to exit.\n", style="yellow")
            continue
        except EOFError:
            break


# ══════════════════════════════════════════════════════════════════════════════
# Single-query mode (pass query as CLI argument)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode: python main.py "explain activity 2 chapter 3 biology"
        single_query = " ".join(sys.argv[1:])

        retriever  = Retriever()
        generator  = Generator()
        tts_engine = _init_tts()

        run_pipeline(single_query, retriever, generator, tts_engine)
    else:
        # Interactive REPL mode: python main.py
        main()