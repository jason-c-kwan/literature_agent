# Literature Search & Triage Agent – Design Specification

## 1 Purpose

Create an AutoGen‑based command‑line assistant that conducts exhaustive literature searches, triages results for relevance and quality, and returns a cited Markdown report plus exportable bibliographic files (RIS) and PDFs. The agent is optimised for life‑science queries (metagenomics, natural‑products chemistry, etc.), leverages only open‑access APIs at prototype stage, and runs locally with Google Gemini 2.5 Pro as the sole LLM.

## 2 Functional Workflow

1. **Receive query** (CLI argument or stdin).
2. **Clarify query** – Query‑RefinerAgent may ask the user follow‑up questions; refined Boolean/MeSH strings are produced.
3. **Search APIs in parallel** – SearchToolAgent calls: NCBI Entrez E‑utilities, Europe PMC, Semantic Scholar, Crossref, OpenAlex, Unpaywall.
4. **First‑pass triage** – TriageAgent scores title & abstract (LLM relevance) and filters by journal quality, OA status, and age‑normalised citation count.
5. **Full‑text retrieval & deep triage** – PDFs or XML are pulled; similarity retrieval (embeddings) selects key passages for a second LLM relevance pass.
6. **Ranking** – RankerAgent combines LLM score (50 %), z‑scored citation rate (30 %), and journal SJR percentile (20 %).
7. **Summarise** – SummariserAgent writes a structured Markdown report with numerically indexed citations.
8. **Export** – ExporterAgent writes `report.md` and `selected.ris`, then ensures PDFs for *all* numerically cited papers are fetched (via Europe PMC, Unpaywall, or fallback scraping) and saved in `output/run-YYYYMMDD/pdfs/` using the naming pattern `[citation‑number]_<FirstAuthor>_<Year>.pdf`. A convenience archive `output/run-YYYYMMDD.zip` containing the same directory tree is also produced.

## 3 System Architecture

### 3.1 Agents

| Agent                  | Back‑end        | Key Responsibilities                                |
| ---------------------- | --------------- | --------------------------------------------------- |
| **UserProxyAgent**     | stdin/stdout    | Receives original question/input parameters.        |
| **Query‑RefinerAgent** | Gemini          | Clarify, expand synonyms, generate search strings.  |
| **SearchToolAgent**    | Python function | Asynchronous API calls + de‑duplication (DOI/PMID). |
| **TriageAgent**        | Gemini          | First‑pass title/abstract relevance assessment.     |
| **RankerAgent**        | Python & Gemini | Combine metrics into final score.                   |
| **SummariserAgent**    | Gemini          | Compose human‑readable report.                      |
| **ExporterAgent**      | Python function | Build RIS, zip PDFs, write report.                  |

### 3.2 External APIs & Fields Used

| API              | Purpose               | Key Fields                                                   |
| ---------------- | --------------------- | ------------------------------------------------------------ |
| NCBI E‑utilities | PubMed search         | PMID, title, abstract, MeSH, journal.                        |
| Europe PMC       | OA full text          | PDF/HTML link, grant info.                                   |
| Semantic Scholar | Citations & influence | `citationCount`, `influentialCitationCount`, `isOpenAccess`. |
| Crossref         | Metadata fallback     | Journal ISSN, licence.                                       |
| OpenAlex         | Journal SJR & metrics | `cited_by_count`, `SJR_percentile`.                          |
| Unpaywall        | OA PDF lookup         | `oa_status`, `best_oa_location->url_for_pdf`.                |

### 3.3 Ranking Formula

```
score = 0.5 * LLM_relevance + 0.3 * norm_citation + 0.2 * SJR_percentile
```

All inputs are Min‑Max normalised to 0–1.

## 4 Command‑Line Interface

```bash
python -m cli.litsearch "query string" \
  --out report.md               # path to Markdown
  --max-papers 250              # cap per‑API results
  --no-pdf                      # skip full‑text download
  --debug                       # TRACE logger output
```

The CLI streams agent dialogue to stdout, diagnostics to stderr; only the final report and export artefacts are written to disk.

\## 5 Repository Layout

```
literature-agent/
├── autogen/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── registry.py
│   └── workflows/
│       └── litsearch.py
├── cli/
│   └── litsearch.py
├── tools/
│   ├── search.py        # async API wrappers
│   ├── triage.py        # abstract/title LLM calls
│   ├── ranking.py       # metric fusion logic
│   └── export.py        # RIS & zip builder
├── config/
│   ├── agents.yaml      # AutoGen agent definitions
│   ├── api_keys.yaml    # API credentials & rate limits
│   └── settings.yaml    # global weights, flags
├── tests/
│   ├── test_search.py
│   ├── test_triage.py
│   └── test_cli.py
├── data/
│   └── cache/           # local JSON/PDF cache
├── docs/
│   └── architecture.md
├── .env_example         # template for env vars
├── requirements.txt
└── README.md
```

## 6 Configuration & Secrets

* `.env` for `GOOGLE_API_KEY`, `NCBI_EMAIL`, etc.
* `config/api_keys.yaml` for non‑secret IDs (cross‑checked into repo if safe).
* `settings.yaml` exposes tunable parameters (weights, max results, vector‑store backend).

## 7 Dependencies

* `autogen>=0.4.0`
* `httpx`, `pydantic`, `tenacity` (retry)
* `python-dotenv`, `rich` (CLI colour)
* `rispy`, `reportlab` (RIS + PDF)
* `chromadb` or `faiss-cpu` for embeddings cache

## 8 Future Enhancements

* **Citation chaining** via Semantic Scholar referenced/citing endpoints.
* **Vector store caching** keyed by DOI hash.
* **Batch scheduling** with n8n or GitHub Actions.
* **Streamlit GUI** that calls the same CLI entry‑point.
* **Pan‑disciplinary mode** – plug in arXiv, BioRxiv, ChemRxiv for pre‑prints.

---

*Last revised: 2025‑05‑21*
