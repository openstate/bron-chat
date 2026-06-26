from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from unstructured.documents.elements import (
    CompositeElement,
    Element,
    ElementMetadata,
    NarrativeText,
)

from app.config import settings
from app.partitioning import html_partition, html_txt_partition, md_partition, txt_partition
from app.utils import composite_to_markdown

PATCH = "app.partitioning.{}"


def _make_elements(count):
    elements = []
    for _ in range(count):
        el = MagicMock()
        el.metadata.orig_elements = [MagicMock()]
        elements.append(el)
    return elements


def _patch_helpers(stack, chunks=None, empty=False):
    if chunks is None:
        chunks = ["chunk 1", "chunk 2"]
    stack.enter_context(patch(PATCH.format("suppress_stdout_stderr")))
    mock_md = stack.enter_context(patch(PATCH.format("composite_to_markdown"), side_effect=chunks))
    stack.enter_context(patch(PATCH.format("custom_clean"), side_effect=lambda x: x))
    stack.enter_context(patch(PATCH.format("is_empty"), return_value=empty))
    return mock_md


class TestHtmlPartition:
    def test_returns_chunks_from_elements(self):
        elements = _make_elements(2)
        with ExitStack() as stack:
            _patch_helpers(stack, chunks=["chunk 1", "chunk 2"])
            stack.enter_context(patch(PATCH.format("partition_html"), return_value=elements))
            result = html_partition("<p>hello</p>")
        assert result == ["chunk 1", "chunk 2"]

    def test_filters_empty_chunks(self):
        elements = _make_elements(1)
        with ExitStack() as stack:
            _patch_helpers(stack, chunks=[""], empty=True)
            stack.enter_context(patch(PATCH.format("partition_html"), return_value=elements))
            result = html_partition("<p>hello</p>")
        assert result == []

    def test_exception_returns_empty_list(self):
        with ExitStack() as stack:
            stack.enter_context(patch(PATCH.format("suppress_stdout_stderr")))
            stack.enter_context(patch(PATCH.format("partition_html"), side_effect=Exception("parse error")))
            result = html_partition("<bad>")
        assert result == []


class TestTxtPartition:
    def test_returns_chunks_from_elements(self):
        elements = _make_elements(2)
        with ExitStack() as stack:
            _patch_helpers(stack, chunks=["chunk 1", "chunk 2"])
            stack.enter_context(patch(PATCH.format("partition_text"), return_value=elements))
            result = txt_partition("plain text")
        assert result == ["chunk 1", "chunk 2"]

    def test_exception_returns_empty_list(self):
        with ExitStack() as stack:
            stack.enter_context(patch(PATCH.format("suppress_stdout_stderr")))
            stack.enter_context(patch(PATCH.format("partition_text"), side_effect=Exception("fail")))
            result = txt_partition("text")
        assert result == []


class TestMdPartition:
    def test_returns_chunks_from_elements(self):
        elements = _make_elements(2)
        with ExitStack() as stack:
            _patch_helpers(stack, chunks=["chunk 1", "chunk 2"])
            stack.enter_context(patch(PATCH.format("partition_md"), return_value=elements))
            result = md_partition("# heading\n\n**field** value")
        assert result == ["chunk 1", "chunk 2"]

    def test_exception_returns_empty_list(self):
        with ExitStack() as stack:
            stack.enter_context(patch(PATCH.format("suppress_stdout_stderr")))
            stack.enter_context(patch(PATCH.format("partition_md"), side_effect=Exception("fail")))
            result = md_partition("# heading")
        assert result == []


class TestHtmlTxtPartition:
    def test_strips_html_tags_before_partitioning(self):
        captured = {}

        def fake_partition(text, **kwargs):
            captured["text"] = text
            return []

        with ExitStack() as stack:
            _patch_helpers(stack, chunks=[])
            stack.enter_context(patch(PATCH.format("partition_text"), side_effect=fake_partition))
            html_txt_partition("<html><body><p>Hello</p><p>World</p></body></html>")

        assert "<p>" not in captured["text"]
        assert "<html>" not in captured["text"]
        assert "<body>" not in captured["text"]
        assert "Hello" in captured["text"]

    def test_returns_chunks_from_elements(self):
        elements = _make_elements(1)
        with ExitStack() as stack:
            _patch_helpers(stack, chunks=["chunk 1"])
            stack.enter_context(patch(PATCH.format("partition_text"), return_value=elements))
            result = html_txt_partition("<p>text</p>")
        assert result == ["chunk 1"]

    def test_exception_returns_empty_list(self):
        with ExitStack() as stack:
            stack.enter_context(patch(PATCH.format("suppress_stdout_stderr")))
            stack.enter_context(patch(PATCH.format("partition_text"), side_effect=Exception("fail")))
            result = html_txt_partition("<p>text</p>")
        assert result == []


class TestCompositeToMarkdown:
    """Pieces of a text-split oversized element must use their own (split) text,
    not a re-render of the full original element."""

    def _composite(self, text, orig_texts):
        orig: list[Element] = [NarrativeText(text=t) for t in orig_texts]
        return CompositeElement(text=text, metadata=ElementMetadata(orig_elements=orig))

    def test_combined_chunk_renders_orig_elements(self):
        composite = self._composite("alpha\n\nbeta", ["alpha", "beta"])
        assert composite_to_markdown(composite) == "alpha\n\nbeta\n\n"

    def test_split_chunk_falls_back_to_own_text(self):
        full = "word " * 700  # one oversized element
        piece = full[:995]
        composite = self._composite(piece, [full])
        assert composite_to_markdown(composite) == piece

    def test_split_fallback_strips_page_numbers(self):
        full = "tekst " * 700
        piece = "Pagina 3 van 10 " + full[:500]
        composite = self._composite(piece, [full])
        assert "Pagina 3 van 10" not in composite_to_markdown(composite)

    def test_missing_orig_elements_uses_own_text(self):
        composite = CompositeElement(text="just text", metadata=ElementMetadata())
        assert composite_to_markdown(composite) == "just text"


class TestChunkSizeRegression:
    def test_oversized_elements_produce_bounded_unique_chunks(self):
        """End-to-end through real unstructured chunking: a document with
        oversized paragraphs must yield chunks near MAX_CHARACTERS, with no
        duplicated content across chunks (the re-expansion bug)."""
        # 3 distinct paragraphs x ~4000 chars, every token unique so any
        # duplicated chunk text can only come from the re-expansion bug.
        doc = "\n\n".join(
            " ".join(f"woord{p}n{i}" for i in range(450)) for p in range(3)
        )
        chunks = txt_partition(doc)

        assert len(chunks) > 3  # oversized paragraphs were split
        assert all(len(c) <= settings.MAX_CHARACTERS + 200 for c in chunks)
        assert len(set(chunks)) == len(chunks)  # no duplicate chunks
