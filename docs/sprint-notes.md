# Flic-a-Disc: 30-Day Sprint — "The Discogs Application Build"

**Window:** 2026-06-18 → 2026-07-17
**Mission:** Turn an already-sophisticated personal tool into a *must-hire* portfolio piece for a Discogs engineering role — by shipping the three things that prove I understand their business better than most candidates:

1. **A working commerce platform** (the Flic-a-Disc Marketplace from the methodology) — cost-plus transparent fees, real payments, seller payouts. This is "I have built the thing you sell, end to end."
2. **Automated daily scrape → recommend pipelines** (eBay + Discogs) — the system runs itself and produces a fresh buy-list every morning. This is "I build production data systems, not demos."
3. **A genuinely better ML recommender** — margin-aware, evaluated honestly, tracked. This is "I do real ML, and I tie it to a P&L."

> **North-star framing.** Discogs makes money on an 11.2% fee. The methodology's entire thesis is that *loving a record correlates with knowing its market*, so high-confidence model picks are also the widest ask-vs-market spreads — the path from **27% → 40% margin**. Every line of this sprint should ladder up to one sentence in the cover letter: *"I built a marketplace, a daily buy-recommendation pipeline, and a margin-aware recommender, and here are the numbers."*

---

## Assessment (where we actually are, 2026-06-18)

### Genuinely strong — lead with these
- **Neural contextual bandit** (`ml/bandit/neural_bandit.py`) with a **contrastive encoder** (`contrastive_encoder.py`), Thompson sampling, and uncertainty estimates. Trained, persisted, online-updatable. Not toy ML.
- **12 catalog classifier tiers** (`ml/catalog/t1..t12/`) — LightGBM + master_id/label propagation, API-calibrated thresholds, ~210 records/API-call coverage on a 2.9M catalog.
- **Knapsack purchase optimizer** (OR-Tools, `views.py:888`) with weight tuning from pairwise ranking — *directly* the "help a buyer spend a budget at a record fair" use case.
- **Three-service production architecture**: Go (Gin) API, Django ML service, React/TS frontend, PostgreSQL, deployed on a DigitalOcean droplet.
- **Real Discogs OAuth + rate-limited scraping at scale**, the seller-inventory "loophole" for cheap labels, and a working `DiscogsBySeller` browser on a built TanStack `DataTable`.

### What hurts the application right now
- **No automation.** Every scrape/recommend is on-demand via an HTTP trigger. There is no daily pipeline — the headline goal #2 literally does not exist yet (no cron, no scheduler, no persisted daily output). `backend` confirms: "No cron/scheduler found."
- **eBay recommend is a 2.5-stage stub.** `ebay.go:370` Stage 3 ("call ML with full features") is a TODO; it currently returns Stage 1 scores. The eBay model training loop is also a TODO (`train_ebay_model.py`).
- **Dead routes.** `DiscogsKeepers`, `DiscogsInventoryView`, `DiscogsSellerTrigger`, `EbayBuyItNow`, `EbayKeepers`, `WfmuPlaylistParser`, `TradingSimulator` are empty divs. A reviewer clicking the nav hits blank pages — instant credibility loss.
- **The marketplace doesn't exist.** The methodology's platform vision (Helcim, cost-plus fees, payouts, seller onboarding) is 0% built. This is the single biggest differentiator and the biggest lift.
- **Honesty gaps:** ~800 `print()` debug statements and scattered `console.log`s signal unfinished work. JWT had a hardcoded default (recent commit `bb68a7d` removed a credential — verify the auth path is clean). Test coverage is thin (only eBay handler/client).
- **No offline ML evaluation actually run.** The `HANDOVER.md` test-set/evaluation framework was specced but `test_set.json` and `evaluate_pipeline.py` were never finished. We claim 92% precision from OOF — we've never verified it on held-out data. A Discogs ML reviewer *will* ask.

### Strategy
ML depth first — because the recommender's margin signal is exactly what the daily pipeline ranks and filters on, so it has to exist before the cron jobs are worth building. Then a week automating the daily scrape→recommend on top of it, then the marketplace, then one week turning it all into a submission. **Aggressive but honest** — every claim in the final writeup is something we can demo or show a number for. We do *not* paper over gaps; we close the ones that matter and explicitly document the ones we don't.

---

## The 30 days at a glance

| Week | Theme | Headline deliverable |
|------|-------|----------------------|
| **1 (D1–7)** | **Make the ML genuinely better** | Margin-aware recommender, real held-out eval harness (precision/recall/coverage), MLflow + Optuna, single champion catalog model — the signal the daily pipeline will rank on |
| **2 (D8–14)** | Harden + **automate the daily pipeline** | A cron-driven daily eBay+Discogs scrape→recommend that persists a fresh margin-filtered buy-list every morning; all dead routes either built or removed |
| **3 (D15–23)** | **The Flic-a-Disc Marketplace** | Seller onboarding → listing → checkout → Helcim payment → cost-plus fee calc → payout, with a live "your effective rate vs Discogs 11.2%" dashboard |
| **4 (D24–30)** | Productionize + **the application package** | CI/CD, monitoring, security, test push, README + architecture diagram + screenshots + demo video + case-study writeup |

**Daily rhythm:** Morning = the hard/novel build. Afternoon = wiring, tests, cleanup. End of day = commit, update this checklist, jot one sentence for the writeup. Target 70%+ completion = success; momentum over perfection.

---

# WEEK 1 — TABLE IMPROVEMENTS

> *Application thesis: "The recommender isn't just a classifier. It's margin-aware, honestly evaluated on held-out data, tracked in MLflow, and tuned with Optuna. Here are the precision/recall/coverage numbers — verified, not OOF-optimistic." This is also the signal the Week-2 daily pipeline ranks and filters on, so it has to be real first.*

- implement annotation everywhere you can. right now, by-seller is a prime example. you've got the code for this elsewhere, just gotta find it
- rewire and refine the annotation/prediction framework
- work to train + refine bandit until it can predict 90%. and maybe it should be perplexity, not accuracy? 
- annotate database Records with their catalog tier and then archive all of that code
- finish editing free shippers
- rewire suggested price to scoring, and look into easing embeddings back in
- finish the training view that gives you a list of records ranked and you re-rank them to tune weights
- revisit to-dos. is it useful?
- revisit the dash in total. is IT useful? you don't use it. perhaps that should get folded into the Catalog.tsx too
- buy it now implementation
- auction implementation


### Day 4 — Off-policy / counterfactual bandit evaluation
**Morning**
- [ ] Implement off-policy evaluation for the bandit (Inverse Propensity Scoring or a simple replay estimator) using the logged annotation history (`BanditTrainingInstance`). This answers "is the bandit's policy actually better than random/greedy?" — a genuinely senior ML move.
- [ ] Log the estimated policy value over model versions.

**Afternoon**
- [ ] Add model-version comparison to `/bandit/metrics/` and the dashboard chart: precision/recall + estimated policy value per version over time. Drift visible at a glance.

### Day 5 — MLflow tracking + Optuna tuning
**Morning**
- [ ] Wire MLflow (already a dep) around training: log hyperparams, the eval-harness metrics, calibration, and policy-value for everey run. `mlruns/` becomes a real experiment log.

**Afternoon**
- [ ] Optuna sweep over the bandit's key hyperparams (embedding dims, dropout, loss weights 0.8/0.2 classification/triplet, learning rate). Pick the best by held-out precision-at-coverage, retrain the champion, log to MLflow.
- [ ] Screenshot the MLflow runs + Optuna parallel-coordinates plot for the writeup.

### Day 6 — Production ML hygiene
**Morning**
- [ ] Add a basic **drift check**: if today's score distribution diverges from the training distribution beyond a threshold, flag it (log it now; route it to the Discord alert once that webhook lands in Week 2). Model monitoring in prod = strong application signal.

**Afternoon**
- [ ] Add Django tests for the margin re-ranking, calibration, and the eval harness.

### Day 7 — Week-1 checkpoint
**Morning**
- [ ] Re-run the full eval harness on the tuned champion. Record final numbers.
- [ ] Write `docs/ML.md`: architecture (bandit + contrastive encoder + Thompson sampling), the catalog pipeline + ablation, the margin-aware re-ranking, the eval methodology, off-policy eval, MLflow/Optuna, drift monitoring. **This doc is half the ML interview.**

**Afternoon — checkpoint**
- [ ] **Writeup milestone:** "Recommender: margin-aware, calibrated, held-out precision X% at Y records/API-call, off-policy value +Z% over greedy, tracked in MLflow." Numbers, not adjectives.

---

# WEEK 2 — Harden the base & automate the daily pipeline (D8–14)

> *Application thesis: "The system runs itself. Every morning it produces a ranked buy-list across eBay and Discogs, filtered to a 40% margin floor — using the Week-1 margin-aware recommender as the ranking signal."*

### Day 8 — Kill the dead routes; make the nav honest
**Morning**
- [ ] Decide per dead route: **build** or **delete**. Build: `DiscogsKeepers`, `EbayKeepers` (these are the buy-list outputs — high value). Delete: `WfmuPlaylistParser`, `TradingSimulator`, `DiscogsSellerTrigger` (out of scope for this sprint) — remove routes from `App.tsx` and nav.
- [ ] Build `DiscogsKeepers.tsx` on the existing `DataTable` — paginated `wanted=true` records from `GET /api/discogs/keepers`. Columns: artist, title, label, year, wants, haves, suggested_price, **spread** (suggested − ask), **margin%** (from the Week-1 recommender payload). Sortable by margin desc by default.

**Afternoon**
- [ ] Add a search/filter bar (artist) to the keepers table.
- [ ] Strip `console.log` spam: `DiscogsKnapsack.tsx:61`, `DiscogsTraining.tsx:42/75/112/163`, `DiscogsCatalogTraining.tsx:42/74/111/163`, `App.tsx:45`.
- [ ] Commit. Writeup note: "Every nav item now resolves to a working view."

### Day 9 — Complete eBay recommend Stage 3 (the real ML ranking)
**Morning**
- [ ] Implement `ebay.go:370` Stage 3: take the Stage-2 enriched top-200 (full metadata) and POST to a new ML endpoint `/ml/ebay/rank/` that returns the **full-feature** keeper score (not the Stage-1 TF-IDF pass).
- [ ] In Django, implement `/ml/ebay/rank/` — run enriched eBay listings through the bandit feature extractor (parse artist/album/label/condition from title via `enhance_listings.py`) and the Week-1 champion model. Return score + uncertainty + projected margin.

**Afternoon**
- [ ] Implement the eBay model **training loop** (`train_ebay_model.py` TODO) — train on annotated eBay listings (`EbayBatchPerformance` + labeled `EbayListing`), persist as a `BanditModel` variant or a dedicated eBay head.
- [ ] Verify end-to-end: `GET /api/ebay/recommend` now returns full-feature ranked results. Add a Go test asserting Stage 3 is hit, plus a Django test for `/ml/ebay/rank/`.

### Day 10 — Persist scrape output (stop living in memory)
**Morning**
- [ ] Today the eBay cache is in-memory (`EbayHandler.cachedListings`). Persist every scrape to `ebay_listings` with a `scraped_at` timestamp and `recommend_score`. This is the substrate the daily pipeline writes to.
- [ ] Add a `scrape_run` table (Go model + migration): `id, source (ebay_auction|ebay_bin|discogs_seller), started_at, finished_at, items_found, items_recommended, status, error`. Every pipeline run logs here — this becomes the monitoring story.

**Afternoon**
- [ ] Build `EbayKeepers.tsx` — the eBay equivalent of the keepers table, reading persisted recommended listings, sorted by score, with bid/BIN/end-time columns.
- [ ] Implement `GetListingRate` persistence so the BIN listing-rate estimate is stored daily (trend over time = a nice chart later).

### Day 11 — The daily scrape scheduler
**Morning**
- [ ] Build a scheduler. Pick the simplest robust option: a Go background worker using `robfig/cron` started in `cmd/api/main.go` (single process, already deployed) **or** a separate `cmd/scheduler` binary + systemd timer on the droplet. Decision: **`cmd/scheduler` binary + systemd timer** — clean separation, survives API restarts, easy to reason about. Document the choice.
- [ ] Job 1 — **eBay daily**: 06:00 ET, scrape last-24h auctions + new BIN, run the full 3-stage recommend, persist top-N with scores, write a `scrape_run` row.

**Afternoon**
- [ ] Job 2 — **Discogs seller sweep**: iterate `sellers_low_minimum.json` (foreign free-shipping sellers — the primary channel), fetch inventory via the seller-loophole, enrich+score, apply the **demand floor** auto-reject (from `catalog-todo.txt` Seller Session spec), persist candidates above a margin threshold.
- [ ] Job 3 — wire the **nightly retrain** + drift-check built in Week 1 (Day 6) into the scheduler, running after the scrape jobs; route drift alerts to Discord.
- [ ] Make all jobs idempotent + rate-limit-respecting (reuse `rate_limiter.py` / the 200ms eBay page sleep). Guard against partial-failure (log to `scrape_run`, continue).

### Day 12 — The morning buy-list (the deliverable that sells the pipeline)
**Morning**
- [ ] Build a "Today's Picks" digest: after the daily jobs finish, assemble the top recommendations across both channels, **filtered to ≥40% projected margin** (the methodology's buying filter, using the Week-1 margin signal), ranked by spread × model confidence.
- [ ] Surface it at `/dashboard` as the hero panel ("Today's Buy-List — N picks, $X projected margin") and persist a snapshot so the dashboard shows *this morning's* list, not a live recompute.

**Afternoon**
- [ ] Add a notification channel (Discord webhook or email via the droplet) that posts the digest each morning. Even if I'm the only recipient, "the system pings me a buy-list at 6am" is a great demo line. (This is the webhook the Week-1 drift check routes to.)
- [ ] Wire a manual "Run now" button (admin-only) that triggers the same pipeline for demos.

### Day 13 — Margin math everywhere (make the methodology visible)
**Morning**
- [ ] Centralize the margin formula in one place (Go + Python share the constants — `BATCH_SIZE`, fee rates, margin target, demand floor in a config/constants module per the sprint.md medium to-do). Compute: net proceeds = price × (1 − discogs_fee); margin = (net − buy_price)/net; spread = suggested − ask.
- [ ] Ensure `projected_margin` and `spread` are on every recommendation payload (eBay + Discogs keepers + knapsack items) — consistent with the Week-1 recommender.

**Afternoon**
- [ ] Knapsack: change the objective to be **margin-aware** — maximize total projected margin under budget, not just score. This is a small but powerful tie-in: the optimizer now directly serves 27%→40%.
- [ ] Update `DiscogsKnapsack.tsx` to show projected total margin alongside total score/cost.

### Day 14 — Week-2 cleanup & checkpoint
**Morning**
- [ ] Strip the ~800 `print()` statements in `ml/bandit/` — replace the handful of useful ones with `logging.info`, delete the rest. Same for `fmt.Println` in `discogs_knapsack.go`.
- [ ] Fix bare `except:` blocks in `catalog/t1/t2/t3/t11/pipeline.py` — at minimum log the exception.
- [ ] Verify JWT is env-only and fails loudly if missing (`auth.go`); audit `.gitignore` for `.env`, `discogs_token.json`, `sellers.json`, credentials.

**Afternoon — checkpoint**
- [ ] Demo the daily pipeline end to end: trigger it, watch `scrape_run`, see the buy-list appear, get the Discord ping.
- [ ] **Writeup milestone:** screenshot the morning digest + a `scrape_run` history. One paragraph: "Daily automated buy-list pipeline, eBay + Discogs, margin-filtered."

---

# WEEK 3 — The Flic-a-Disc Marketplace (D15–23)

> *Application thesis: "I didn't just analyze a marketplace — I built one. Cost-plus transparent fees, real card payments, seller payouts, and a live dashboard showing each seller they're paying ~3% instead of Discogs' 11.2%."*

This is the biggest lift and the biggest differentiator. Build database-outward (per the `catalog-todo.txt` bulk-tool discipline): prove the money math in isolation before any UI can move money.

### Day 15 — Domain model & money math (no UI)
**Morning**
- [ ] Design the schema (Go models + migrations, Django mirror): `SellerAccount` (payout method, Discogs handle, stats), `MarketplaceListing` (record FK, condition, price, shipping, status), `Order` (buyer, line items, totals), `Transaction` (gross, fee breakdown, net), `Payout` (seller, amount, status, period), `OverheadLedger` (DO hosting + stipend + tooling, by month).
- [ ] Implement the **cost-plus fee engine** as a pure, unit-tested function: `fee(transaction) = helcim(1.79% + $0.08) + overhead(total_overhead / rolling_4wk_txn_count)`. Shipping is pass-through, never platform revenue.

**Afternoon**
- [ ] Unit-test the fee engine hard: the worked example from the methodology (1,000 txns/mo, $50 avg, $600 overhead → ~$1.575/txn ≈ 3.15%) must reproduce exactly. Test the rolling-window recalculation as volume changes.
- [ ] Guard: never charge a negative/zero fee; overhead divisor floor; shipping isolated from the fee base.

### Day 16 — Seller onboarding
**Morning**
- [ ] `POST /api/marketplace/sellers` — create a seller account: handle, payout method (PayPal/ACH), agree-to-terms. Auth-gated.
- [ ] Seller dashboard page: account status, listings, sales, payouts, and the **live effective-rate widget** (their rolling rate vs Discogs 11.2%, with $ saved).

**Afternoon**
- [ ] Optional but high-value: **import listings from a Discogs seller inventory** — reuse the existing seller-inventory scraper to pre-populate `MarketplaceListing` drafts. "Switch from Discogs in one click" is a killer onboarding story and reuses code we already have.

### Day 17 — Listings & marketplace browse
**Morning**
- [ ] `POST/PATCH/DELETE /api/marketplace/listings` — seller CRUD on listings (price, condition, shipping, qty).
- [ ] Public browse/search endpoint + page. Use Postgres full-text search (artist/title/label) — fast, no new infra. Filter by genre/style/condition/price.

**Afternoon**
- [ ] **Recommender-powered discovery:** rank/feature browse results with the Week-1 recommender so the marketplace surfaces desirable records first. This is the moment the ML and the platform become one system — call it out explicitly in the writeup.
- [ ] Listing detail page: condition, seller, shipping, "add to cart."

### Day 18 — Cart & checkout
**Morning**
- [ ] Cart model + endpoints (add/remove/update qty). Multi-seller cart splits into per-seller orders (each seller ships separately — shipping is per-seller, per the methodology).
- [ ] Checkout page: order summary, per-seller shipping, **transparent fee line shown to the buyer-facing total and the seller-facing net**.

**Afternoon**
- [ ] Order creation: lock prices, compute fees via the engine, create `Order` + `Transaction` rows in `pending` state. No money moves yet.

### Day 19 — Helcim payment integration
**Morning**
- [ ] Integrate Helcim (HelcimPay.js hosted fields or their API) — **never touch raw card data**; use their tokenization/hosted checkout for PCI scope minimization. Document this decision (security-aware = hireable).
- [ ] On successful charge: mark `Transaction` paid, capture the real processing fee from Helcim's response, finalize the fee breakdown.

**Afternoon**
- [ ] Webhook/callback handling for async payment status; idempotency keys so a double-callback can't double-charge or double-fulfill (mirror the bulk-tool "never run twice" discipline).
- [ ] Use Helcim's sandbox/test mode for all of this — demo with test cards, document that production keys are env-only.

### Day 20 — Order lifecycle & payouts
**Morning**
- [ ] Order state machine: `pending → paid → fulfilled → completed` (+ `cancelled`/`refunded`). Seller marks fulfilled (tracking optional). Buyer/auto-complete after a window.
- [ ] Payout calculation: on completion, seller's net (gross − fee − shipping-passthrough-to-seller... actually shipping flows *to* seller) accrues to a `Payout`.

**Afternoon**
- [ ] Payout batch job (add to scheduler): weekly, aggregate completed orders per seller into a `Payout`, mark for PayPal/ACH disbursement. (Actual disbursement can be manual/logged for the demo — document it.)
- [ ] Seller payout history view.

### Day 21 — The transparency dashboard (the recruiter screenshot)
**Morning**
- [ ] Build the **fee-transparency dashboard**: live overhead ledger, rolling 4-week transaction count, current effective rate, and a per-seller "you saved $X vs Discogs this period" figure. This single screen is the most persuasive artifact in the whole application.
- [ ] Admin view: total volume, overhead coverage, progress toward the $2,000/mo stipend target (~4,000 txns/mo across ~40 sellers at $50 = $200k GMV).

**Afternoon**
- [ ] Seller-recruitment incentive hook (per the methodology — "sellers have direct incentive to recruit other sellers"): a referral field on signup + a "more sellers = lower everyone's rate" explainer on the dashboard. Even just modeled/visualized, it shows I understand the platform's growth flywheel.

### Day 22 — Marketplace hardening
**Morning**
- [ ] Money-path integration tests: full happy path (browse → cart → checkout → pay → fulfill → payout) against Helcim sandbox; refund path; multi-seller split; idempotent webhook replay.
- [ ] Authorization audit: a seller can only edit their own listings/orders; buyers can't see other buyers' orders; admin-only endpoints locked down.

**Afternoon**
- [ ] Edge cases: out-of-stock at checkout, price-changed-since-cart, payment failure rollback (no order left in a bad state), concurrent purchase of the last copy.

### Day 23 — Week-3 checkpoint
**Morning**
- [ ] End-to-end demo: create a seller, import a few Discogs listings, buy one with a Helcim test card, fulfill, see the fee breakdown and the "vs Discogs" savings.
- [ ] Seed realistic demo data (a handful of sellers, ~50 listings) so screenshots and the demo video look alive, not empty.

**Afternoon — checkpoint**
- [ ] **Writeup milestone:** screenshots of checkout with the transparent fee line + the transparency dashboard. One paragraph: "Built a working co-op marketplace — Helcim payments, cost-plus fees recomputed on a rolling 4-week window, seller payouts. ~3% effective vs Discogs' 11.2%."

---

# WEEK 4 — Productionize & assemble the application (D24–30)

> *Application thesis: "It's deployed, tested, monitored, secured — and packaged so a reviewer understands it in five minutes without reading code."*

### Day 24 — CI/CD
**Morning**
- [ ] `.github/workflows/ci.yml` — on every push/PR: `go build ./... && go vet ./... && go test ./...`, `npm ci && npm run build && tsc --noEmit`, `python manage.py check && pytest`.
- [ ] Fix `deploy.sh` Go binary path if still wrong (`./cmd/server` → `./cmd/api`) and account for the new `cmd/scheduler`.

**Afternoon**
- [ ] Deploy job on push to `main` after CI passes: SSH to droplet, pull, rebuild all services + scheduler timer, migrate, restart. Droplet SSH key as a GH Actions secret.
- [ ] Green CI badge in the README.

### Day 25 — Containerization & deploy verification
**Morning**
- [ ] Dockerfiles for all three services + the scheduler (multi-stage). `docker-compose.yml` for the full local stack. (The `frontend/Dockerfile` already exists — extend the pattern.)
- [ ] Deploy and verify all services + the systemd scheduler timer come up clean; run migrations including all new marketplace + scrape_run tables.

**Afternoon**
- [ ] Smoke-test every nav item on `flic-a-disc.com` — nothing 404s, nothing's an empty div. Confirm SSE/long-poll endpoints aren't buffered by nginx if used.
- [ ] Confirm all env vars set on droplet: DB, JWT_SECRET, ML_SERVICE_URL, EBAY_APP_ID/CERT_ID, HELCIM keys (sandbox), Discord webhook.

### Day 26 — Monitoring & alerting
**Morning**
- [ ] Prometheus + Grafana on the droplet (or DO managed). Instrument Go + Django: request latency/error rate, scrape-run success/duration, daily-pick count, marketplace GMV/txn count.
- [ ] Dashboard panels: pipeline health, ML metrics over time, marketplace volume vs the stipend target.

**Afternoon**
- [ ] Alerting: scrape job failed, ML drift flagged, payment webhook error, service down → Discord. The `scrape_run` table makes most of this trivial.

### Day 27 — Security hardening
**Morning**
- [ ] Secrets audit end-to-end (git history too — `bb68a7d` removed a credential; confirm nothing else is committed; rotate anything that was). Move all secrets to env/secret store, fail loudly if missing.
- [ ] Payment-path review: confirm no raw card data ever hits our servers (Helcim hosted/tokenized), HTTPS everywhere, security headers, rate-limit auth + payment endpoints.

**Afternoon**
- [ ] Run `/security-review` over the diff; fix findings. Add authz tests for the money path.

### Day 28 — Test coverage push
**Morning**
- [ ] Go: tests for `LabelCatalogRecords`, `GetWantedRecords`, the eBay Stage-3 path, the fee engine, the order state machine.
- [ ] Django: tests for `catalog_candidates`, `/ml/ebay/rank/`, margin re-ranking, the eval harness.

**Afternoon**
- [ ] Frontend: at least smoke/component tests for checkout and the keepers tables. Target a coverage number worth quoting (~60%+ on the money + pipeline paths).
- [ ] `go vet ./...`, `tsc --noEmit`, `pytest` all green.

### Day 29 — The application package: README, diagram, screenshots
**Morning**
- [ ] Write the top-level `README.md`: what it is, the methodology in two sentences, the three pillars (marketplace / daily pipeline / margin-aware ML), tech stack, **architecture diagram** (Go ↔ Django ML ↔ React ↔ Postgres, scheduler, Helcim, Discogs/eBay scrapers), and a "5-minute tour" with screenshots.
- [ ] Curate screenshots: morning buy-list, keepers table with margin, knapsack optimizer, MLflow runs, calibration plot, checkout with transparent fee, transparency dashboard ("vs Discogs 11.2%").

**Afternoon**
- [ ] Record a 3–5 min demo video: trigger the daily pipeline → buy-list → annotate → marketplace checkout → fee transparency. A reviewer watching this is 80% to "interview."
- [ ] Update the in-app tour (`TourView`) to walk the *now-working* features incl. marketplace + buy-list.

### Day 30 — The narrative & final pass
**Morning**
- [ ] Write `docs/CASE_STUDY.md` (the cover-letter spine): the problem (27%→40% margin, 11.2% fees), what I built, the architecture decisions and trade-offs, the real ML numbers, the marketplace economics, what I'd do next. Honest about gaps (payout disbursement is logged-not-automated in the demo; coverage where it's thin).
- [ ] Final full-app pass: click every nav item, run the pipeline, complete a marketplace purchase, confirm nothing crashes.

**Afternoon — ship**
- [ ] Tag a release. Confirm CI green, droplet healthy, demo data seeded.
- [ ] Assemble the submission: repo link, README, CASE_STUDY, demo video, the ML.md + the eval numbers. Done.

---

## Anything I might be missing (deliberately scoped IN, and OUT)

**Scoped in (because they make the application stronger):**
- **Margin-aware ranking** — the single best idea to tie ML to the business; it's the bridge between the methodology and the recommender (Week 1 / Day 3).
- **Off-policy bandit evaluation** — separates a senior ML candidate from a "trained a model" candidate (Week 1 / Day 4).
- **Recommender-powered marketplace discovery** — makes the two big pillars one system (Day 17).
- **Fee-transparency dashboard with "vs Discogs"** — the most persuasive single screen for *this specific employer* (Day 21).
- **Drift monitoring + MLflow + Optuna** — "I run ML in production," not "I trained a notebook" (Week 1 / Days 5–6).

**Scoped out (and why — say so in the case study, don't fake it):**
- **Real payout disbursement / full KYC** — demo uses logged payouts + Helcim sandbox. Document as "production would add Helcim/PayPal payout API + identity verification."
- **WFMU parser / trading simulator** — removed; not relevant to the Discogs story.
- **Mobile-native** — responsive web only; note it.

## Risks & escape valves
- **Marketplace week is the riskiest.** If Helcim integration stalls, fall back to a fully-tested fee engine + order lifecycle with a **mocked payment provider** behind a clean interface — the *economics and transparency* are the point, and a swappable `PaymentProvider` interface is itself a good design signal. Don't let payments block the dashboard.
- **If ML eval reveals the 92% claim was inflated** — good. Report the real number and the methodology. Honesty + a working harness beats an unverifiable boast every time.
- **70% rule:** if a day slips, carry the build task and protect the *checkpoint demos* (D7, D14, D23) — those are what the writeup is built from.
- **Daily discipline:** end every day with a commit + one writeup sentence. The case study should assemble itself from 30 sentences.

## Definition of done (the must-hire bar)
- [ ] Every nav item resolves to a working feature.
- [ ] A scheduled daily pipeline produces a margin-filtered buy-list across eBay + Discogs, with run history.
- [ ] The recommender is margin-aware, calibrated, and has **real held-out numbers** + off-policy eval, tracked in MLflow.
- [ ] A buyer can complete a real (sandbox) purchase; the seller sees a transparent cost-plus fee and an accruing payout; the dashboard shows ~3% vs Discogs' 11.2%.
- [ ] CI green, deployed, monitored, secrets clean.
- [ ] README + architecture diagram + screenshots + 3–5 min demo video + case study — a reviewer gets it in five minutes.
</content>
</invoke>
