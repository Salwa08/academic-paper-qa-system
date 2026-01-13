import arxiv
import os
from pathlib import Path
import json
import shutil
import subprocess

TEMP_PDF_DIR = Path("data/temp_pdfs")
PROCESSED_DIR = Path("data/processed")
TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PAPERS=[
    {"id": "1706.03762", "title": "Attention is All You Need"},
    {"id": "1810.04805", "title": "BERT: Pre-training..."},  
    {"id": "1810.13083", "title": "GraphIE: A Graph-Based Framework for Information Extraction"},
]

client = arxiv.Client(page_size=50, delay_seconds=3.1)

def download_paper(arxiv_id: str) -> Path:
    pdf_path = TEMP_PDF_DIR / f"{arxiv_id}.pdf"
    
    # Check if already extracted first
    md_path = PROCESSED_DIR / arxiv_id / f"{arxiv_id}.md"
    if md_path.exists():
        print(f"Already extracted {arxiv_id}, skipping download.")
        return pdf_path
    
    if not pdf_path.exists():
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(client.results(search))
        print(f"Downloading {result.title}...")
        result.download_pdf(dirpath=str(TEMP_PDF_DIR), filename=f"{arxiv_id}.pdf")
    else:
        print(f"Already have {arxiv_id}, skipping download.")

    return pdf_path


def extract_to_md(pdf_path: Path, arxiv_id: str) -> Path:
    output_dir = PROCESSED_DIR / arxiv_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {arxiv_id} with marker...")
    subprocess.run([
        "marker_single",
        str(pdf_path),
        "--output_format", "markdown",
        "--output_dir", str(output_dir),
    ], check=True)

    md_path = output_dir / f"{arxiv_id}.md"
    if md_path.exists():
        print(f"Extraction complete: {md_path}")
        return md_path
    else:
        raise FileNotFoundError(f"Markdown file not found after extraction: {md_path}")
    


metadata = []

for paper in PAPERS:
    try:
        pdf_path = download_paper(paper["id"])
        md_path = extract_to_md(pdf_path, paper["id"])

        pdf_path.unlink(missing_ok=True)
        print(f"Cleaned up temporary PDF: {pdf_path}")

        metadata.append({
            "id": paper["id"],
            "title": paper["title"],
            "markdown_path": str(md_path),
            "extracted_at": str(Path(md_path).stat().st_mtime),
        })

    except Exception as e:
        print(f"Error processing paper {paper['id']}: {e}")

with open(PROCESSED_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Ugly MVP extraction done! Raw PDFs cleaned up. Ready for indexing.")