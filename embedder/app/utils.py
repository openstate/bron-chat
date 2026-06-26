import sys
import os
import contextlib
import re
import logging
from lxml import etree
from unstructured.cleaners.core import clean_non_ascii_chars, replace_unicode_quotes

logger = logging.getLogger(__name__)

PAGINA_PATTERN_1 = r'Pagina \d+ van \d+'


def is_empty(text: str) -> bool:
    return len(text.strip()) == 0


def has_page_numbers(text: str) -> bool:
    return bool(re.search(PAGINA_PATTERN_1, text))


def element_to_markdown(element) -> str:
    if has_page_numbers(element.text):
        return ""

    text = ""
    if element.category == 'Title':
        depth = element.metadata.category_depth if element.metadata.category_depth is not None else 1
        heading_level = min(6, depth + 2)
        text = f"{'#' * heading_level} {element.text}\n\n"
    elif element.category in ['Header', 'SubHeader']:
        heading_level = 2 if element.category == 'Header' else 3
        text = f"{'#' * heading_level} {element.text}\n\n"
    elif element.category == 'ListItem':
        text = f"- {element.text}\n"
    elif element.category in ['Paragraph', 'NarrativeText', 'CompositeElement', 'UncategorizedText', 'Table']:
        text = f"{element.text}\n\n"
    else:
        text = f"{element.text}\n"

    return text


def elements_to_markdown(elements) -> str:
    return "".join([element_to_markdown(el) for el in elements])


def composite_to_markdown(composite_element) -> str:
    """Markdown for one chunk produced by unstructured's chunking.

    Renders orig_elements to preserve structure, except for pieces of a
    text-split oversized element: there orig_elements holds the full pre-split
    element, and rendering it would duplicate the element into every piece."""
    orig_elements = composite_element.metadata.orig_elements
    if not orig_elements:
        return composite_element.text
    orig_chars = sum(len(el.text) for el in orig_elements)
    if orig_chars > len(composite_element.text):
        return re.sub(PAGINA_PATTERN_1, "", composite_element.text)
    return elements_to_markdown(orig_elements)


def custom_clean(text: str) -> str:
    text = clean_non_ascii_chars(text)
    text = text.replace("•\n", "")
    text = text.replace("- \n", "- ")
    text = replace_unicode_quotes(text)
    text = text.rstrip("\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def remove_processing_instructions(html_text: str) -> str:
    try:
        with suppress_stdout_stderr():
            parser = etree.HTMLParser(remove_pis=True)
            tree = etree.fromstring(html_text.encode('utf-8'), parser)
            return etree.tostring(tree, encoding='unicode', method='html')
    except Exception:
        return html_text
