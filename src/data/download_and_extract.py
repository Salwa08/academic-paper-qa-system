import arxiv
import os
from pathlib import Path
import json
import shutil
import subprocess
import time
from typing import Optional

# Use absolute paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TEMP_PDF_DIR = PROJECT_ROOT / "data" / "temp_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PAPERS=[
    {"id": "1706.03762", "title": "Attention is All You Need"},
    {"id": "1810.04805", "title": "BERT: Pre-training..."},  
    {"id": "1810.13083", "title": "GraphIE: A Graph-Based Framework for Information Extraction"},
]

# Configuration
USE_FALLBACK = False   # Disable fallback, use only marker-pdf

client = arxiv.Client(page_size=50, delay_seconds=3.1)

def download_paper(arxiv_id: str) -> Path:
    pdf_path = TEMP_PDF_DIR / f"{arxiv_id}.pdf"
    
    # Check if already extracted - look for nested structure
    md_path_nested = PROCESSED_DIR / arxiv_id / arxiv_id / f"{arxiv_id}.md"
    md_path_direct = PROCESSED_DIR / arxiv_id / f"{arxiv_id}.md"
    
    if md_path_nested.exists() or md_path_direct.exists():
        print(f"✓ Already extracted {arxiv_id}, skipping download.")
        return pdf_path
    
    if not pdf_path.exists():
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(client.results(search))
        print(f"⬇️  Downloading {result.title}...")
        result.download_pdf(dirpath=str(TEMP_PDF_DIR), filename=f"{arxiv_id}.pdf")
        print(f"✓ Downloaded to {pdf_path}")
    else:
        print(f"✓ PDF already exists: {arxiv_id}")

    return pdf_path


def extract_with_marker(pdf_path: Path, arxiv_id: str) -> Optional[Path]:
    """Extract PDF using marker-pdf (no timeout, shows real-time output)."""
    output_dir = PROCESSED_DIR / arxiv_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Extracting {arxiv_id} with marker-pdf...")
    print(f"   This may take 5-10 minutes per paper...")
    print(f"   Watching output (Ctrl+C to cancel)...")
    print()
    
    try:
        start_time = time.time()
        
        # Run without timeout and show output in real-time
        result = subprocess.run([
            "marker_single",
            str(pdf_path),
            "--output_format", "markdown",
            "--output_dir", str(output_dir),
        ], check=False)  # Don't raise on non-zero, we'll check manually
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n⚠️  Marker failed with return code {result.returncode}")
            return None

        # Marker creates nested directory: arxiv_id/arxiv_id/arxiv_id.md
        # Try both locations
        md_path_nested = output_dir / arxiv_id / f"{arxiv_id}.md"
        md_path_direct = output_dir / f"{arxiv_id}.md"
        
        if md_path_nested.exists():
            print(f"\n✓ Found markdown at: {md_path_nested}")
            print(f"   Extraction complete in {elapsed:.1f}s")
            return md_path_nested
        elif md_path_direct.exists():
            print(f"\n✓ Found markdown at: {md_path_direct}")
            print(f"   Extraction complete in {elapsed:.1f}s")
            return md_path_direct
        else:
            print(f"\n⚠️  Marker completed but no markdown file found")
            print(f"   Checked: {md_path_nested}")
            print(f"   Checked: {md_path_direct}")
            print(f"   Directory contents:")
            if output_dir.exists():
                for item in output_dir.rglob("*"):
                    print(f"      {item}")
            return None
            
    except FileNotFoundError:
        print(f"\n⚠️  marker_single command not found. Install with: pip install marker-pdf")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction cancelled by user")
        return None
    except Exception as e:
        print(f"\n⚠️  Marker extraction failed: {e}")
        return None


def extract_with_pdfplumber(pdf_path: Path, arxiv_id: str) -> Optional[Path]:
    """Fallback: Extract PDF using pdfplumber (simpler, faster)."""
    try:
        import pdfplumber
    except ImportError:
        print(f"⚠️  pdfplumber not installed. Install with: pip install pdfplumber")
        return None
    
    output_dir = PROCESSED_DIR / arxiv_id
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{arxiv_id}.md"
    
    print(f"📄 Extracting {arxiv_id} with pdfplumber (fallback)...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Paper: {arxiv_id}\n\n")
                f.write(full_text)
            
            print(f"✓ Fallback extraction complete: {md_path}")
            return md_path
            
    except Exception as e:
        print(f"⚠️  Pdfplumber extraction failed: {e}")
        return None


def extract_to_md(pdf_path: Path, arxiv_id: str) -> Path:
    """Extract PDF to markdown with fallback support."""
    
    # Try marker-pdf first
    md_path = extract_with_marker(pdf_path, arxiv_id)
    
    # Fallback to pdfplumber if marker fails and fallback is enabled
    if md_path is None and USE_FALLBACK:
        print(f"🔄 Trying fallback extraction method...")
        md_path = extract_with_pdfplumber(pdf_path, arxiv_id)
    
    if md_path is None:
        raise FileNotFoundError(f"All extraction methods failed for {arxiv_id}")
    
    return md_path
    


if __name__ == "__main__":
    print("\n" + "="*70)
    print("📚 PAPER EXTRACTION SCRIPT")
    print("="*70)
    print(f"Papers to process: {len(PAPERS)}")
    print(f"Output directory: {PROCESSED_DIR}")
    print(f"Fallback enabled: {USE_FALLBACK}")
    print("="*70 + "\n")
    
    metadata = []
    successful = 0
    failed = 0
    skipped = 0

    for i, paper in enumerate(PAPERS, 1):
        print(f"\n{'='*70}")
        print(f"Processing paper {i}/{len(PAPERS)}: {paper['title']}")
        print(f"ArXiv ID: {paper['id']}")
        print('='*70)
        
        try:
            # Check if already fully processed - look for both nested and direct paths
            md_path_nested = PROCESSED_DIR / paper["id"] / paper["id"] / f"{paper['id']}.md"
            md_path_direct = PROCESSED_DIR / paper["id"] / f"{paper['id']}.md"
            existing_md = md_path_nested if md_path_nested.exists() else (md_path_direct if md_path_direct.exists() else None)
            
            if existing_md:
                print(f"✓ Paper already processed at: {existing_md}")
                print(f"  Adding to metadata...")
                metadata.append({
                    "id": paper["id"],
                    "title": paper["title"],
                    "markdown_path": str(existing_md),
                    "extracted_at": str(existing_md.stat().st_mtime),
                })
                skipped += 1
                continue
            
            # Download and extract
            pdf_path = download_paper(paper["id"])
            md_path = extract_to_md(pdf_path, paper["id"])

            # Clean up PDF
            pdf_path.unlink(missing_ok=True)
            print(f"🗑️  Cleaned up temporary PDF: {pdf_path}")

            metadata.append({
                "id": paper["id"],
                "title": paper["title"],
                "markdown_path": str(md_path),
                "extracted_at": str(md_path.stat().st_mtime),
            })
            
            successful += 1
            print(f"✅ Successfully processed {paper['id']}")

        except Exception as e:
            failed += 1
            print(f"❌ Error processing paper {paper['id']}: {e}")
            print(f"   Continuing with remaining papers...")

    # Save metadata
    print(f"\n{'='*70}")
    print("💾 Saving metadata...")
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 EXTRACTION SUMMARY")
    print('='*70)
    print(f"✅ Successful: {successful}")
    print(f"⏭️  Skipped (already processed): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Total in metadata: {len(metadata)}")
    print('='*70)
    
    if len(metadata) > 0:
        print(f"\n✨ Ready for indexing! Run paper_processor.py to build BM25 index.")
    else:
        print(f"\n⚠️  No papers were successfully extracted.")
    
    print()