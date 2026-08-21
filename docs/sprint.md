# Flic-a-Disc: 2-Week Sprint Plan
**Goal: Job application to Discogs — clean, complete, presentable**

---

## Assessment

### What's impressive
- The ML pipeline and data infrastructure is genuinely sophisticated — 12 pipelines, vote bucketing, bandit model, 3M record DB. This is not toy work.
- Knapsack optimizer is directly relevant to Discogs's core use case (helping buyers at record fairs)
- Full OAuth Discogs integration at scale shows deep platform knowledge
- Three-service architecture (Go + Django + React) is production-grade
- RAG system with streaming SSE, ChromaDB, Ollama is a strong differentiator

### What hurts the application right now
- Keepers page is an empty div — the feature that shows wantlist output is invisible
- Debug logs everywhere (`print()`, `fmt.Println`, `console.log`) signal unfinished work
- Three Knapsack session endpoints return `[]` stubs
- No README with screenshots — a reviewer shouldn't have to read code to understand what this does
- Tour exists but needs to walk through working features

---

## What's Already Built (just needs wiring)

| Component | Status | Location |
|-----------|--------|----------|
| RAG Frontend | Complete | `frontend/src/pages/rag/` |
| RAG Django Backend | Integrated ✓ | `ml/rag/` |
| RAG Go Proxy | Integrated ✓ | `backend/internal/handlers/rag.go` |
| Keepers Backend | Complete | `backend/internal/handlers/discogs_keepers.go` |
| Keepers Frontend | Empty stub | `frontend/src/pages/discogs/DiscogsKeepers.tsx` |
| Table Library | Not installed | needs `@tanstack/react-table` + shadcn |

---

## Sprint

### Day 1: RAG wiring ✓
- [x] Copy RAG handler to `backend/internal/handlers/rag.go`
- [x] Register RAG routes in `main.go` (`/api/rag/*path`)
- [x] Copy RAG Django app to `ml/rag/`, register URLs
- [x] Fix relative imports, install `chromadb` + `langchain-text-splitters`
- [x] Run `rag` migrations
- [x] Add RAG settings to `config/settings.py`

### Day 2: RAG corpus + FAQ backend
- [ ] Pull Ollama models: `ollama pull llama3.2:3b && ollama pull nomic-embed-text`
- [ ] Drop conversation `.txt` files into `ml/rag/docs/` (subdirs: `claude/`, `chatgpt/`), run incremental scan from the existing Ingest UI
- [ ] Add `faq_view` (GET/POST) to `rag/views.py` — stores pairs as `{id, question, answer, annotation}` in `ml/rag/faq.json`
- [ ] Add `faq_detail_view` (DELETE) to `rag/views.py`
- [ ] Register both in `rag/urls.py`: `faq/` and `faq/<str:pair_id>/`

### Day 3: RAG frontend rebuild
- [ ] Add `FaqPair` interface + `getFaq`, `saveFaqPair`, `deleteFaqPair` to `api.ts`
- [ ] Update `RagLayout.tsx` — replace existing 5 tabs with `dev` / `faq` / `readme`
- [ ] Write `RagDev.tsx` — question input → streaming answer into editable textarea → annotation field → save / discard buttons
- [ ] Write `RagFAQ.tsx` — loads saved pairs, renders question + answer + annotation, delete button per pair
- [ ] Update `App.tsx` — replace five `/rag/*` routes with `/rag/dev`, `/rag/faq`, `/rag/readme`

### Day 4: Keepers backend + install table library
- [ ] Verify `/api/discogs/wanted` returns the right shape — check `discogs_keepers.go` and add any missing fields (artist, title, label, year, wants, haves, suggested_price)
- [ ] Install TanStack Table: `npm install @tanstack/react-table`
- [ ] Init shadcn: `npx shadcn-ui@latest init` then `npx shadcn-ui@latest add table button input`

### Day 5: Keepers frontend
- [ ] Build `DiscogsKeepers.tsx` — paginated TanStack table of `wanted=True` records
- [ ] Columns: artist, title, label, year, wants, haves, suggested price — sortable by any
- [ ] Add search/filter bar (artist name)
- [ ] Add to nav and dashboard

### Day 6: Table upgrade — EbayAuctions + DiscogsCatalogCandidates
- [ ] Set up shared shadcn `DataTable` pattern (one reusable component)
- [ ] Migrate `EbayAuctions` table to TanStack — column sorting, sticky header, row hover
- [ ] Migrate `DiscogsCatalogCandidates` — preserve shift-click selection

### Day 7: Table upgrade — DiscogsKnapsack + debug cleanup
- [ ] Migrate `DiscogsKnapsack` table to TanStack
- [ ] Strip ~50 bare `print()` calls in `ml/bandit/views.py` — replace key ones with `logging.info`, delete the rest
- [ ] Remove `fmt.Println` debug spam from `backend/internal/handlers/discogs_knapsack.go`
- [ ] Remove `console.log` statements from `frontend/src/pages/discogs/DiscogsCatalogTraining.tsx`
- [ ] Fix JWT hardcoded default in `auth.go` — env-only, fail loudly if missing

### Day 8: Finish the stubs
- [ ] Implement `knapsack_sessions_list` in Django — return `KnapsackSession.objects.all()`
- [ ] Implement `knapsack_session_update` — update notes/saved flag
- [ ] Wire `DiscogsKnapsackComparison.tsx` to real session data
- [ ] eBay `ebay.go:370` Stage 3 TODO — complete or remove and document as known gap

### Day 9: Catalog candidates polish + InventoryView
- [ ] Add progress bar to `DiscogsCatalogCandidates` (annotated / total six-vote records)
- [ ] Add "skip batch" button copy when 0 records selected
- [ ] `DiscogsInventoryView.tsx` — implement as browsable view of `is_six` records, or remove the route entirely

### Day 10: README
- [ ] Write `README.md` — what it is, what it does, tech stack, architecture diagram
- [ ] Add "What this does" section to `BUILD.md` above setup instructions
- [ ] Clean up `results.md` — document which pipelines ran and why t5/t7/t8/t9/t12 are excluded

### Day 11: Screenshots + RAG FAQ generation
- [ ] Take screenshots: annotation UI, knapsack optimizer, keepers page, eBay auctions
- [ ] Run RAG locally, generate FAQ answers for 40-50 questions covering each major feature
- [ ] Review and annotate answers, save to `faq.json`
- [ ] Write `docs/RAG.md` — what the RAG is, what corpus it was trained on, how to run locally, link to FAQ as sample output

### Day 12: GitHub Actions CI/CD
- [ ] Fix `deploy.sh` Go binary path: `./cmd/server` → `./cmd/api`
- [ ] Write `.github/workflows/ci.yml`:
  - **CI job** (every push + PR): `go build ./...`, `go vet ./...`, `go test ./...`, `npm ci && npm run build`, `python manage.py check`
  - **Deploy job** (push to `main` only, after CI passes): SSH into droplet, pull, rebuild, restart
- [ ] Add droplet SSH private key as GitHub Actions secret (`DROPLET_SSH_KEY`)
- [ ] Add green CI badge to `README.md`

### Day 13: Droplet readiness
- [ ] Run deploy and verify all three services restart cleanly
- [ ] Smoke test every nav item on `flic-a-disc.com` — nothing 404s or shows empty
- [ ] Confirm all env vars set on droplet: `DJANGO_URL`, `JWT_SECRET`, `OLLAMA_BASE_URL`, etc.
- [ ] Verify `uv run python manage.py migrate` runs clean on droplet (rag migration 0001 is new)
- [ ] Check nginx SSE buffering is off for `/api/rag/query/` (`proxy_buffering off`)

### Day 14: Tests + tour + final pass
- [ ] Add Go tests for `LabelCatalogRecords` and `GetWantedRecords`
- [ ] Add Django test for `catalog_candidates` GET/POST
- [ ] Run `go vet ./...` and `tsc --noEmit`, fix anything flagged
- [ ] Update tour stops to reflect working features: keepers, RAG FAQ, knapsack, annotation UI
- [ ] Final pass: load the app, click every nav item, confirm nothing 404s or crashes

---

## Code Quality To-Do (do these alongside the sprint, not as a separate pass)

### Critical
- [ ] `auth.go:40` — remove hardcoded JWT default, fail on missing env var
- [ ] Audit `.gitignore` — confirm no `.env`, `discogs_token.json`, or credentials are committed

### High
- [ ] `views.py` — replace bare `print()` with `logging` or delete
- [ ] `discogs_knapsack.go` — remove debug `fmt.Println` calls
- [ ] `DiscogsCatalogTraining.tsx` — remove debug `console.log` calls
- [ ] `knapsack_sessions_list/update/compare` — implement or remove
- [ ] eBay `ebay.go:370` TODO — complete Stage 3 or document as known gap

### Medium
- [ ] `sweep_sellers.py` — remove unused `import random`
- [ ] `DiscogsInventoryView.tsx` — 74 bytes, implement or remove route
- [ ] Add `BATCH_SIZE` and other magic numbers to a config/constants file
- [ ] `results.md` — document missing pipelines (t5, t7, t8, t9, t12)

---

## Table Library

TanStack Table (`@tanstack/react-table`) with shadcn's `DataTable` component. Install:

```bash
npm install @tanstack/react-table
npx shadcn-ui@latest init
npx shadcn-ui@latest add table button input
```

Gives sorting, filtering, pagination, and row selection as composable hooks. Existing shift-click selection logic maps directly onto TanStack's `rowSelection` state. One install, one pattern, replaces all 9 hand-rolled tables.

---

## Submission Notes

**Lead with:**
- The ML pipeline (12 classifiers, vote bucketing, Thompson sampling bandit) — not toy ML
- Knapsack optimizer — directly relevant to Discogs's buyer use case
- Scale: 3M record DB, real OAuth Discogs integration, seller inventory sweep
- Domain expertise — you understand their data model better than most users

**Be upfront about:**
- Test coverage is light — next priority after features are complete
- Some endpoints were stubs during active development (knapsack sessions)
- Debug logging was left in during data pipeline work, being cleaned up
