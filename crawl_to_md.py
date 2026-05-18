#!/usr/bin/env python3
"""
Render each URL listed in a text file (one URL per line) with a headless
browser and save its main content as a Markdown file. No crawling — only
the URLs in the input file are fetched. Pages are fetched in parallel.

Setup:
    pip install playwright markdownify beautifulsoup4
    playwright install chromium

Usage:
    python crawl_to_md.py urls.txt --out ./out --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify
from playwright.async_api import TimeoutError as PWTimeout
from playwright.async_api import async_playwright


# Tags whose content is page chrome, not page content.
# Match on semantic HTML5 tags and ARIA roles only — never class/id, since
# CMS / framework class names are unstable and not part of the page contract.
SELECTORS_TO_STRIP = [
    "button", "script", "style", "noscript", "iframe", "svg", "template",
    "nav", "header", "footer", "aside",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]",
    "[role=dialog]", "[role=alertdialog]",
    "[aria-label*='cookie' i]", "[aria-label*='consent' i]",
    ".modal-root",
]


SAFE_FILENAME = re.compile(r"[^a-zA-Z0-9._-]+")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def url_to_filename(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path.strip("/") or "index"
    if parsed.query:
        path = f"{path}__{parsed.query}"
    slug = SAFE_FILENAME.sub("-", path).strip("-")
    return (slug[:180] or "index") + ".md"


def normalize(url: str) -> str:
    """Strip fragment, normalize trailing slash, drop common tracking params."""
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme.startswith("http"):
        return ""
    query_pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith(("utm_", "gclid", "fbclid"))
    ]
    query = urllib.parse.urlencode(query_pairs)
    path = parsed.path or "/"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc.lower(), path, query, ""))


async def render_page(page, url: str, timeout_ms: int) -> str:
    """Navigate to URL, wait for SPA to settle, return rendered HTML."""
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except PWTimeout:
        pass  # some pages keep long-poll connections open; render is usually done
    return await page.content()


def extract_metadata(soup: BeautifulSoup) -> dict:
    def meta_content(name: str | None = None, prop: str | None = None) -> str:
        attrs = {"name": name} if name else {"property": prop}
        tag = soup.find("meta", attrs=attrs)
        return (tag.get("content") or "").strip() if tag else ""

    title = (
        meta_content(prop="og:title")
        or (soup.title.string.strip() if soup.title and soup.title.string else "")
    )
    description = meta_content(prop="og:description") or meta_content(name="description")
    return {"title": title, "description": description}


def extract_markdown(html: str) -> tuple[str | None, dict]:
    """Strip page chrome, then convert remaining DOM to Markdown."""
    soup = BeautifulSoup(html, "html.parser")
    metadata = extract_metadata(soup)
    body = soup.body or soup
    for selector in SELECTORS_TO_STRIP:
        for element in body.select(selector):
            element.decompose()
    # Collapse repeated empty wrappers that markdownify would turn into noise.
    for element in body.find_all(True):
        if not element.get_text(strip=True) and not element.find(["img", "br", "hr"]):
            element.decompose()

    markdown = markdownify(
        str(body),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    # Squash 3+ consecutive blank lines that markdownify leaves behind.
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return (markdown or None), metadata


def write_page(out_dir: Path, url: str, markdown: str, metadata: dict) -> Path:
    title = (metadata.get("title") or "").replace("\n", " ").strip()
    description = (metadata.get("description") or "").replace("\n", " ").strip()
    frontmatter = [
        "---",
        f"url: {url}",
        f"title: {title}",
        f"description: {description}",
        f"fetched_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    if title:
        frontmatter.extend([f"# {title}", ""])
    path = out_dir / url_to_filename(url)
    path.write_text("\n".join(frontmatter) + markdown.strip() + "\n", encoding="utf-8")
    return path


def merge_pages(out_dir: Path, merged_name: str = "_all.md") -> Path | None:
    page_files = sorted(p for p in out_dir.glob("*.md") if p.name != merged_name)
    if not page_files:
        return None
    merged_path = out_dir / merged_name
    page_contents = [page.read_text(encoding="utf-8").rstrip() for page in page_files]
    merged_path.write_text("\n\n".join(page_contents) + "\n", encoding="utf-8")
    for page in page_files:
        page.unlink()
    return merged_path


def read_urls(path: Path) -> list[str]:
    """One URL per line; blank lines and lines starting with # are ignored."""
    urls = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = normalize(line)
        if normalized:
            urls.append(normalized)
    return urls


async def _block_heavy_resources(route):
    if route.request.resource_type in {"image", "media", "font"}:
        await route.abort()
    else:
        await route.continue_()


async def worker(worker_id: int, browser_context, queue: asyncio.Queue,
                 out_dir: Path, timeout_ms: int, total: int,
                 counters: dict) -> None:
    page = await browser_context.new_page()
    await page.route("**/*", _block_heavy_resources)
    try:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                url_index, url = item
                try:
                    html = await render_page(page, url, timeout_ms)
                except Exception as exc:
                    print(f"[err] {url}: {exc}", file=sys.stderr)
                    counters["skipped"] += 1
                    continue

                markdown, metadata = extract_markdown(html)
                if not markdown or len(markdown.strip()) < 50:
                    print(f"[skip] no content: {url}", file=sys.stderr)
                    counters["skipped"] += 1
                else:
                    path = write_page(out_dir, url, markdown, metadata)
                    counters["fetched"] += 1
                    print(f"[ok {url_index}/{total} w{worker_id}] {url} -> {path.name}",
                          file=sys.stderr)
            finally:
                queue.task_done()
    finally:
        await page.close()


async def run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.md"):
        stale.unlink()

    urls = read_urls(Path(args.urls_file))
    if not urls:
        print(f"[err] no URLs in {args.urls_file}", file=sys.stderr)
        return 1
    total = len(urls)
    concurrency = max(1, min(args.concurrency, total))
    counters = {"fetched": 0, "skipped": 0}

    queue: asyncio.Queue = asyncio.Queue()
    for url_index, url in enumerate(urls, 1):
        queue.put_nowait((url_index, url))
    for _ in range(concurrency):
        queue.put_nowait(None)  # sentinel per worker

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headed)
        browser_context = await browser.new_context(user_agent=USER_AGENT, locale="hu-HU")
        workers = [
            asyncio.create_task(
                worker(worker_id, browser_context, queue, out_dir,
                       args.timeout, total, counters)
            )
            for worker_id in range(1, concurrency + 1)
        ]
        await asyncio.gather(*workers)
        await browser.close()

    merged = merge_pages(out_dir)
    if merged:
        size_kb = merged.stat().st_size / 1024
        print(f"[merge] {merged} ({size_kb:.0f} KB)", file=sys.stderr)

    print(f"[done] wrote {counters['fetched']} files to {out_dir}, "
          f"skipped {counters['skipped']}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls_file", help="Path to text file with one URL per line")
    parser.add_argument("--out", default="./out", help="Output directory (default: ./out)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max parallel page fetches (default: 4)")
    parser.add_argument("--timeout", type=int, default=20000,
                        help="Per-page timeout in ms (default: 20000)")
    parser.add_argument("--headed", action="store_true",
                        help="Show browser window (default: headless)")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
