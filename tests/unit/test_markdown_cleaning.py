from src.data.chunking import SectionAwareChunker


def test_clean_md_removes_images_and_page_spans():
    chunker = SectionAwareChunker()

    text = (
        '# Introduction\n'
        'Some content.\n'
        '![Figure](figure.png)\n'
        '<span id="page-1-2"></span>'
    )

    result = chunker.clean_md(text)

    assert 'Some content.' in result
    assert '![Figure]' not in result
    assert '<span' not in result


def test_clean_md_preserves_regular_text():
    chunker = SectionAwareChunker()

    text = 'This is a normal paragraph.'

    assert chunker.clean_md(text) == text