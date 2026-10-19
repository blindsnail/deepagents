#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

"""
Content Builder Agent

A content writer agent configured entirely through files on disk:
- AGENTS.md defines brand voice and style guide
- skills/ provides specialized workflows (blog posts, social media)
- skills/*/scripts/ provides tools bundled with each skill
- subagents handle research and other delegated tasks

Usage:
    uv run python content_writer.py "Write a blog post about AI agents"
    uv run python content_writer.py "Create a LinkedIn post about prompt engineering"
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Literal

import yaml

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

EXAMPLE_DIR = Path(__file__).parent
console = Console()

# Logging: write structured logs to stderr so they don't compete with the rich
# Console output on stdout. Level can be overridden with the
# CONTENT_WRITER_LOG_LEVEL environment variable (default: INFO).
logger = logging.getLogger("content_writer")
if not logger.handlers:
    _log_level_name = os.environ.get("CONTENT_WRITER_LOG_LEVEL", "INFO").upper()
    _log_level = getattr(logging, _log_level_name, logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
logger.debug("Logger initialised at level %s", _log_level_name)


# Web search tool for the researcher subagent
@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: "general" for most queries, "news" for current events

    Returns:
        Search results with titles, URLs, and content excerpts.
    """
    logger.debug(
        "web_search called: query=%r, max_results=%d, topic=%s",
        query,
        max_results,
        topic,
    )
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.warning("web_search: TAVILY_API_KEY not set in environment")
            return {"error": "TAVILY_API_KEY not set"}

        client = TavilyClient(api_key=api_key)
        logger.info("web_search: querying Tavily (query=%r)", query)
        result = client.search(query, max_results=max_results, topic=topic)
        # Tavily returns a dict that may include a 'results' list.
        results_count = (
            len(result.get("results"))
            if isinstance(result, dict) and isinstance(result.get("results"), list)
            else "n/a"
        )
        logger.info("web_search: completed (query=%r, results=%s)", query, results_count)
        return result
    except Exception as e:
        logger.exception("web_search failed for query=%r: %s", query, e)
        return {"error": f"Search failed: {e}"}


# Qwen-Image 3.0 text-to-image model served via DashScope's
# MultiModalConversation (multimodal-generation) endpoint. It is a distinct
# API from the legacy ImageSynthesis / wanx text-to-image path; the response
# is read from output.choices[0].message.content[0]["image"].
QWEN_IMAGE_MODEL = "qwen-image-3.0"
# DashScope protocol uses width*height (with '*') — note this differs from
# the OpenAI-compatible mode which uses 'x'. Both sizes are within the
# model's 512x512..2048x2048 pixel-area range.
# 16:9 landscape for blog hero images; gives negative space for title overlay,
# as the blog-post skill recommends. ~0.9M pixels (well under the model's 2M cap).
COVER_IMAGE_SIZE = "1280*720"
# Square works across LinkedIn and Twitter/X feeds per the social-media skill.
SOCIAL_IMAGE_SIZE = "1024*1024"


def _generate_image_with_qwen(prompt: str, output_path: Path, size: str) -> str:
    """Generate one image with the Qwen-Image 3.0 model via DashScope and save
    it to `output_path`. Returns a status string for the @tool caller.
    """
    import urllib.request
    from http import HTTPStatus

    from dashscope import MultiModalConversation

    logger.debug(
        "_generate_image_with_qwen: prompt=%r, size=%s, output=%s",
        prompt,
        size,
        output_path,
    )
    if not os.environ.get("DASHSCOPE_API_KEY"):
        logger.warning("_generate_image_with_qwen: DASHSCOPE_API_KEY not set")
        return "Error: DASHSCOPE_API_KEY not set"

    try:
        logger.info(
            "Calling DashScope MultiModalConversation (model=%s, size=%s)",
            QWEN_IMAGE_MODEL,
            size,
        )
        rsp = MultiModalConversation.call(
            model=QWEN_IMAGE_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            result_format="message",
            n=1,
            size=size,
        )
    except Exception as e:
        logger.exception("_generate_image_with_qwen: DashScope call failed: %s", e)
        return f"Error: {e}"

    if rsp.status_code != HTTPStatus.OK:
        msg = (
            f"Error: Qwen image generation failed "
            f"(status={rsp.status_code}, code={rsp.code}, message={rsp.message})"
        )
        logger.error(
            "Qwen image generation failed: status=%s code=%s message=%s",
            rsp.status_code,
            getattr(rsp, "code", None),
            getattr(rsp, "message", None),
        )
        return msg

    image_url: str | None = None
    for choice in getattr(rsp.output, "choices", None) or []:
        content = getattr(getattr(choice, "message", None), "content", None) or []
        for part in content:
            if isinstance(part, dict) and part.get("image"):
                image_url = part["image"]
                break
        if image_url:
            break
    if not image_url:
        logger.warning("Qwen returned no image url in response")
        return "No image generated"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading generated image to %s", output_path)
        with urllib.request.urlopen(image_url, timeout=120) as resp:
            data = resp.read()
        output_path.write_bytes(data)
        logger.info("Saved image to %s (%d bytes)", output_path, len(data))
    except Exception as e:
        logger.exception("Failed to download generated image to %s: %s", output_path, e)
        return f"Error: failed to download generated image: {e}"
    return f"Image saved to {output_path}"


@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post.

    Args:
        prompt: Detailed description of the image to generate.
        slug: Blog post slug. Image saves to blogs/<slug>/hero.png
    """
    output = EXAMPLE_DIR / "blogs" / slug / "hero.png"
    logger.debug("generate_cover invoked: slug=%s, output=%s", slug, output)
    result = _generate_image_with_qwen(prompt, output, COVER_IMAGE_SIZE)
    logger.debug("generate_cover result: %s", result)
    return result


@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post.

    Args:
        prompt: Detailed description of the image to generate.
        platform: Either "linkedin" or "tweets"
        slug: Post slug. Image saves to <platform>/<slug>/image.png
    """
    output = EXAMPLE_DIR / platform / slug / "image.png"
    logger.debug(
        "generate_social_image invoked: platform=%s, slug=%s, output=%s",
        platform,
        slug,
        output,
    )
    result = _generate_image_with_qwen(prompt, output, SOCIAL_IMAGE_SIZE)
    logger.debug("generate_social_image result: %s", result)
    return result


def load_subagents(config_path: Path) -> list:
    """Load subagent definitions from YAML and wire up tools.

    NOTE: This is a custom utility for this example. Unlike `memory` and `skills`,
    deepagents doesn't natively load subagents from files - they're normally
    defined inline in the create_deep_agent() call. We externalize to YAML here
    to keep configuration separate from code.
    """
    # Map tool names to actual tool objects
    available_tools = {
        "web_search": web_search,
    }

    logger.info("Loading subagent definitions from %s", config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    subagents = []
    for name, spec in config.items():
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        if "model" in spec:
            subagent["model"] = spec["model"]
        if "tools" in spec:
            subagent["tools"] = [available_tools[t] for t in spec["tools"]]
        subagents.append(subagent)
        logger.debug(
            "Loaded subagent %r (model=%s, tools=%s)",
            name,
            subagent.get("model", "<inherited>"),
            [getattr(t, "name", repr(t)) for t in subagent.get("tools", [])],
        )

    logger.info("Loaded %d subagents from %s", len(subagents), config_path)
    return subagents


def create_content_writer():
    """Create a content writer agent configured by filesystem files."""
    subagents = load_subagents(EXAMPLE_DIR / "subagents.yaml")
    logger.info(
        "Creating deep agent: model=deepseek-flash, memory=[AGENTS.md], "
        "skills=[./skills/], tools=[generate_cover, generate_social_image], "
        "subagents=%d, backend_root=%s",
        len(subagents),
        EXAMPLE_DIR,
    )
    agent = create_deep_agent(
        model = "deepseek-flash",
        memory=["./AGENTS.md"],           # Loaded by MemoryMiddleware
        skills=["./skills/"],             # Loaded by SkillsMiddleware
        tools=[generate_cover, generate_social_image],  # Image generation
        subagents=subagents,  # Custom helper
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
    )
    logger.info("Deep agent created successfully")
    return agent


class AgentDisplay:
    """Manages the display of agent progress."""

    def __init__(self):
        self.printed_count = 0
        self.current_status = ""
        self.spinner = Spinner("dots", text="Thinking...")

    def update_status(self, status: str):
        self.current_status = status
        self.spinner = Spinner("dots", text=status)

    def print_message(self, msg):
        """Print a message with nice formatting."""
        kind = type(msg).__name__
        logger.debug("print_message: kind=%s", kind)
        if isinstance(msg, HumanMessage):
            logger.debug("HumanMessage content_len=%d", len(str(msg.content or "")))
            console.print(Panel(str(msg.content), title="You", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = "\n".join(text_parts)

            logger.debug(
                "AIMessage text_len=%d tool_calls=%d",
                len(content or ""),
                len(msg.tool_calls or []),
            )
            if content and content.strip():
                console.print(Panel(Markdown(content), title="Agent", border_style="green"))

            if msg.tool_calls:
                logger.debug(
                    "AIMessage tool_call names=%s",
                    [tc.get("name") for tc in msg.tool_calls],
                )
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    logger.debug("tool_call name=%r args_keys=%s", name, list(args.keys()))

                    if name == "task":
                        desc = args.get("description", "researching...")
                        console.print(f"  [bold magenta]>> Researching:[/] {desc[:60]}...")
                        self.update_status(f"Researching: {desc[:40]}...")
                    elif name in ("generate_cover", "generate_social_image"):
                        console.print(f"  [bold cyan]>> Generating image...[/]")
                        self.update_status("Generating image...")
                    elif name == "write_file":
                        path = args.get("file_path", "file")
                        console.print(f"  [bold yellow]>> Writing:[/] {path}")
                    elif name == "web_search":
                        query = args.get("query", "")
                        console.print(f"  [bold blue]>> Searching:[/] {query[:50]}...")
                        self.update_status(f"Searching: {query[:30]}...")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content_text = str(msg.content or "")
            logger.debug(
                "ToolMessage name=%r content_len=%d",
                name,
                len(content_text),
            )
            if name in ("generate_cover", "generate_social_image"):
                if "saved" in content_text.lower():
                    console.print(f"  [green]✓ Image saved[/]")
                else:
                    console.print(f"  [red]✗ Image failed: {content_text}[/]")
            elif name == "write_file":
                console.print(f"  [green]✓ File written[/]")
            elif name == "task":
                console.print(f"  [green]✓ Research complete[/]")
            elif name == "web_search":
                if "error" not in content_text.lower():
                    console.print(f"  [green]✓ Found results[/]")


async def main():
    """Run the content writer agent with streaming output."""
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "Write a blog post about how AI agents are transforming software development"

    logger.info("Starting content writer run with task=%r", task)
    console.print()
    console.print("[bold blue]Content Builder Agent[/]")
    console.print(f"[dim]Task: {task}[/]")
    console.print()

    agent = create_content_writer()
    display = AgentDisplay()

    console.print()

    # Use Live display for spinner during waiting periods
    logger.info("Streaming agent responses...")
    with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:
        chunk_count = 0
        async for chunk in agent.astream(
            {"messages": [("user", task)]},
            config={"configurable": {"thread_id": "content-writer-demo"}},
            stream_mode="values",
        ):
            chunk_count += 1
            logger.debug(
                "astream chunk #%d received, keys=%s",
                chunk_count,
                sorted(chunk.keys()),
            )
            if "messages" in chunk:
                messages = chunk["messages"]
                if len(messages) > display.printed_count:
                    new_count = len(messages) - display.printed_count
                    logger.debug(
                        "Printing %d new message(s) (total so far: %d)",
                        new_count,
                        len(messages),
                    )
                    # Temporarily stop spinner to print
                    live.stop()
                    for msg in messages[display.printed_count:]:
                        display.print_message(msg)
                    display.printed_count = len(messages)
                    # Resume spinner
                    live.start()
                    live.update(display.spinner)

    logger.info(
        "Agent stream finished; total chunks=%d, total messages=%d",
        chunk_count,
        display.printed_count,
    )
    console.print()
    console.print("[bold green]✓ Done![/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (KeyboardInterrupt)")
        console.print("\n[yellow]Interrupted[/]")
