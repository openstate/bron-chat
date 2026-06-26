import logging
from typing import Callable, List
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.cleaners.core import group_broken_paragraphs

from app.utils import composite_to_markdown, custom_clean, is_empty, suppress_stdout_stderr
from app.config import settings

logger = logging.getLogger(__name__)


def _partition(partition_fn: Callable, doc: str, label: str, **extra_kwargs) -> List[str]:
    markdown_chunks = []
    try:
        with suppress_stdout_stderr():
            elements = partition_fn(
                text=doc,
                chunking_strategy="by_title",
                paragraph_grouper=group_broken_paragraphs,
                combine_text_under_n_chars=settings.COMBINE_TEXT_UNDER_N_CHARS,
                max_characters=settings.MAX_CHARACTERS,
                max_partition=settings.MAX_PARTITION,
                overlap=settings.OVERLAP,
                **extra_kwargs,
            )
        for composite_element in elements:
            markdown = composite_to_markdown(composite_element)
            cleaned_markdown = custom_clean(markdown)
            if not is_empty(cleaned_markdown):
                markdown_chunks.append(cleaned_markdown)
    except Exception as e:
        logger.error(f"Error during {label} partitioning: {e}")
    return markdown_chunks


def html_partition(doc: str) -> List[str]:
    return _partition(partition_html, doc, "HTML")


def txt_partition(doc: str) -> List[str]:
    return _partition(partition_text, doc, "TXT")


def html_txt_partition(doc: str) -> List[str]:
    doc = (
        doc.replace("<p>", "")
        .replace("</p>", "\n\n")
        .replace("<body>", "")
        .replace("</body>", "")
        .replace("<html>", "")
        .replace("</html>", "")
    )
    return _partition(
        partition_text, doc, "HTML-TXT", new_after_n_chars=settings.NEW_AFTER_N_CHARS
    )
