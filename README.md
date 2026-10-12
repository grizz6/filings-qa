# Filings Q&A

**Ask questions about the risks major US companies disclose, and get answers cited to the
exact passages of their SEC filings.**

[![Live app](https://img.shields.io/badge/live%20app-filings.streamlit.app-2a78d6)](https://filings.streamlit.app)
[![CI](https://github.com/grizz6/filings-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/grizz6/filings-qa/actions/workflows/ci.yml)
[![Live app check](https://github.com/grizz6/filings-qa/actions/workflows/live-check.yml/badge.svg)](https://github.com/grizz6/filings-qa/actions/workflows/live-check.yml)

## What's inside

- **Data pipeline:** SEC EDGAR 10-K filings → *Item 1A. Risk Factors* → 400-word chunks →
  `all-MiniLM-L6-v2` embeddings → Supabase Postgres with pgvector
- **Question answering:** company-aware vector search (top 8 passages) → Google Gemini
  writes an answer that cites every passage it uses
- **Web app:** Streamlit (Ask and Monitoring pages) and a FastAPI service, packaged with
  Docker and hosted on Streamlit Community Cloud
- **Quality:** a verified 50-question benchmark, experiments tracked in MLflow, and an
  evaluation gate on every change
- **Operations:** GitHub Actions for testing, deployment, daily live checks and
  monitoring; secrets held in encrypted stores and protected by secret scanning and
  row-level security

## The problem

The *Risk Factors* section of a 10-K is where a company lists everything that could hurt its
business. It runs to dozens of pages of dense legal text, so finding what one company says
about one specific risk means reading the whole section. General-purpose chatbots answer
from memory instead: they can be out of date, cannot show where an answer came from, and
will produce a confident answer even when the filing says nothing on the subject.

## The solution

Filings Q&A answers plain-English questions using only the text of each company's latest
10-K. Every statement in an answer is numbered and linked to the passage it came from, so it
can be checked in seconds. When the filing does not cover a question, the app replies
"Not found in the filings." instead of guessing.

> **Q:** What does Tesla say about supply chain risk?
>
> **A:** Tesla faces risks regarding the availability of components and suppliers, which could
> lead to production delays, idle facilities, and an inability to fulfill customer
> contracts [8]. …
>
> **Source [8]:** TSLA 10-K, Item 1A

**Companies covered:** Apple, Microsoft, Tesla, JPMorgan Chase, Walmart, Pfizer,
Exxon Mobil, Nike, Netflix and Delta Air Lines.

## Results

Measured on 50 questions: 40 with a known answer, each checked word-for-word against the
filing, and 10 that the filings do not answer.

| Setup | Chunk size | Passages retrieved | Right passage found | Ranking (MRR) | Correctly declined |
|---|---|---|---|---|---|
| A | 200 words | 5 | 75% | 0.52 | 10/10 |
| B | 400 words | 5 | 82.5% | 0.60 | 10/10 |
| **Live app** | **400 words** | **8** | **90%** | **0.61** | **10/10** |

## Repository structure

```
src/        Data pipeline, retrieval, answer generation, evaluation, API and monitoring
app/        Streamlit app: Ask page and Monitoring dashboard
eval/       Benchmark questions and experiment settings
checks/     Automated checks of the live app
tests/      Unit tests
.github/    GitHub Actions workflows
```
