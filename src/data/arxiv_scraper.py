# TODO: Implement ArXiv scraper class
# Learn to build a web scraper from scratch!
# 
# Hints for your learning journey:
# 1. Look into the 'arxiv' Python library for easy API access
# 2. Understand how to query the ArXiv API for papers
# 3. Don't forget rate limiting to be respectful to the API
# 4. Think about error handling for network issues
# 5. Structure your data well (JSON format works great)
#
import arxiv
import json
import tqdm
import os
from arxiv import UnexpectedEmptyPageError  

METADATA_FILE = "data/raw/arxiv_metadata.json" 
RAW_DATA_DIR = "data/raw/pdfs/"
os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

categories = ["cs.AI", "cs.CL", "cs.LG"]
query = " OR ".join([f"cat:{cat}" for cat in categories])

client = arxiv.Client(
    page_size=3500,
    delay_seconds=3
)
search = arxiv.Search(
    query=query,
    max_results= 1000,
    # sort_by=arxiv.SortCriterion.SubmittedDate,  # Sort by date
    # sort_order=arxiv.SortOrder.Descending       # Newest first
)
papers = []
downloaded_pdfs = 0

try:
    for result in tqdm.tqdm(client.results(search), total=search.max_results):
        try:
            paper_id = result.entry_id.split("/")[-1]
            if paper_id in [p["ID"] for p in papers]:
                continue

            filename = f"{paper_id}.pdf"
            file_path = os.path.join(RAW_DATA_DIR, filename)

            # Skip if PDF file already exists
            if os.path.exists(file_path):
                tqdm.tqdm.write(f"PDF already exists for paper: {paper_id}, skipping download.")
            else:
                # Download the PDF
                result.download_pdf(dirpath=RAW_DATA_DIR, filename=filename)
                tqdm.tqdm.write(f"Downloaded PDF for paper: {result.title}")
                downloaded_pdfs += 1
            
            # Store Metadata
            paper = {
                "ID": paper_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "pdf_path": file_path
            }
            
            papers.append(paper)
            

        except Exception as e:
            tqdm.tqdm.write(f"Error downloading paper {result.title}: {str(e)}")
            continue
            
except UnexpectedEmptyPageError:
    tqdm.tqdm.write("No more results available - reached the end of available papers.")

except Exception as e:
    tqdm.tqdm.write(f"An error occurred during scraping: {str(e)}")

tqdm.tqdm.write(f"Successfully collected metadata for {len(papers)} papers.")
tqdm.tqdm.write(f"Downloaded {downloaded_pdfs} new PDFs.")

with open(METADATA_FILE, "w", encoding="UTF-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"Total processed papers: {len(papers)} and metadata saved to {METADATA_FILE}")