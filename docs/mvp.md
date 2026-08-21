# **Flic-a-Disc: 3-Month Sprint Plan**

**Aggressive Timeline \- Motivation Over Precision**

---

## **Month 1: Core Features \+ Critical Fixes**

### **Week 1: Knapsack & Debugging**

**Day 1: Redis Setup & Knapsack Design**

* \[ \] Install Redis on droplet  
* \[ \] Configure Redis systemd service  
* \[ \] Test Redis connection from Go backend  
* \[ \] Test Redis connection from Django ML service  
* \[ \] Design knapsack data structure (seller inventory, weights, scores)  
* \[ \] Write knapsack algorithm pseudocode  
* \[ \] Document knapsack scoring function (embedding \+ price\_diff \+ is\_want)

**Day 2: Knapsack Backend (Django)**

* \[ \] Create `/ml/knapsack/optimize/` endpoint  
* \[ \] Implement greedy knapsack solver  
  * \[ \] Pre-filter: remove bottom 50% by score  
  * \[ \] Round prices to nearest dollar  
  * \[ \] Sort by score/price ratio  
  * \[ \] Fill until budget  
* \[ \] Add seller inventory fetching logic  
* \[ \] Test with mock data (10,000 items, $100 budget)  
* \[ \] Verify performance (\<20ms response time)

**Day 3: Knapsack Backend (Go) & Database**

* \[ \] Create `DiscogsSeller` model/table (seller\_id, shipping\_minimum, currency)  
* \[ \] Seed sellers table with JSON data  
* \[ \] Create `/api/knapsack/sellers` endpoint (filter by budget)  
* \[ \] Create `/api/knapsack/inventory` endpoint (get seller's records)  
* \[ \] Create `/api/knapsack/optimize` endpoint (proxy to Django)  
* \[ \] Add knapsack results to database (KnapsackSession table)

**Day 4: Knapsack Frontend**

* \[ \] Create KnapsackPage.tsx component  
* \[ \] Add budget input field  
* \[ \] Add three weight sliders (embedding, price\_diff, is\_want)  
* \[ \] Fetch sellers on budget change  
* \[ \] Display top seller's inventory (50 items)  
* \[ \] Show knapsack dividing line (items above \= selected)  
* \[ \] Add red highlight when over budget  
* \[ \] Add "Purchase" button

**Day 5: Knapsack Polish & Debug**

* \[ \] Implement real-time knapsack re-optimization on slider change (debounce 200ms)  
* \[ \] Add loading states and error handling  
* \[ \] Test edge cases:  
  * \[ \] No valid sellers for budget  
  * \[ \] Seller has \<10 records  
  * \[ \] All records too expensive  
  * \[ \] Weight sliders at extremes (0 or 1\)  
* \[ \] Cache knapsack results in Redis (5 min TTL)

**Day 6: Discogs Trainer Debug**

* \[ \] Review trainer logs for errors  
* \[ \] Test trainer with 100 records locally  
* \[ \] Fix model architecture mismatches  
* \[ \] Fix feature extraction pipeline  
* \[ \] Verify triplet generation working  
* \[ \] Test online learning (trigger after 5 batches)  
* \[ \] Verify threshold auto-tuning

**Day 7: Buffer & Cleanup**

* \[ \] Fix any remaining knapsack bugs  
* \[ \] Fix any remaining trainer bugs  
* \[ \] Write tests for knapsack solver  
* \[ \] Update documentation for new features  
* \[ \] Code cleanup and linting

**Goal:** Knapsack working end-to-end, trainer debugged

---

### **Week 2: Inventory & Keepers Pages**

**Day 8: Keepers Database Model**

* \[ \] Design unified Keeper model (Discogs \+ eBay)  
* \[ \] Add fields: source (discogs/ebay), record\_id, title, artist, price, etc.  
* \[ \] Create database migration  
* \[ \] Write script to populate from existing Record \+ EbayKeeper tables  
* \[ \] Run migration and data population script  
* \[ \] Verify data integrity

**Day 9: Keepers Backend API**

* \[ \] Create `/api/keepers` endpoint (GET all keepers, paginated)  
* \[ \] Create `/api/keepers/discogs` endpoint (filter by source)  
* \[ \] Create `/api/keepers/ebay` endpoint (filter by source)  
* \[ \] Add sorting options (date added, price, artist)  
* \[ \] Add search/filter functionality  
* \[ \] Test with 500+ keeper records

**Day 10: Keepers Inventory Frontend**

* \[ \] Create KeepersPage.tsx component  
* \[ \] Add tabs: "All", "Discogs", "eBay"  
* \[ \] Display keepers in grid layout (similar to annotation view)  
* \[ \] Add pagination (40 items per page)  
* \[ \] Add sorting dropdown  
* \[ \] Add search bar  
* \[ \] Style with existing MySpace theme

**Day 11: eBay Keepers Annotation Interface**

* \[ \] Create EbayKeepersAnnotation.tsx component  
* \[ \] Fetch unannotated eBay keepers  
* \[ \] Display 40 per page with Keep/Skip buttons  
* \[ \] Add "Notes" field for each keeper  
* \[ \] Submit annotations to `/api/ebay/keepers/annotate`  
* \[ \] Update keeper status in database  
* \[ \] Add progress indicator

**Day 12: Merge View**

* \[ \] Add "Merged View" tab to KeepersPage  
* \[ \] Implement de-duplication logic (match Discogs \+ eBay by title/artist)  
* \[ \] Show merged records with source badges  
* \[ \] Add "Link" button to manually connect Discogs \+ eBay records  
* \[ \] Store links in database (keeper\_links table)  
* \[ \] Test merge accuracy

**Day 13: Keepers Stats & Visualization**

* \[ \] Add keepers count to dashboard  
* \[ \] Add breakdown by source (Discogs vs eBay)  
* \[ \] Add total value estimate  
* \[ \] Create simple chart (keepers added over time)  
* \[ \] Add export functionality (CSV download)

**Day 14: Testing & Polish**

* \[ \] Test all keepers pages with large datasets  
* \[ \] Fix pagination bugs  
* \[ \] Fix search/filter bugs  
* \[ \] Add loading skeletons  
* \[ \] Write tests for keepers API  
* \[ \] Code cleanup

**Goal:** Complete inventory management system

---

### **Week 3: BuyItNow \+ Tour**

**Day 15: BuyItNow Trigger Design**

* \[ \] Design BuyItNow monitoring system  
* \[ \] Create BuyItNowListing model (listing\_id, title, price, seller, end\_time)  
* \[ \] Create BuyItNowAlert model (alert\_id, listing\_id, score, triggered\_at)  
* \[ \] Plan cron job vs continuous monitoring approach  
* \[ \] Document trigger conditions (score threshold, price range)

**Day 16: BuyItNow Backend**

* \[ \] Create `/api/ebay/buyitnow/monitor` endpoint  
* \[ \] Implement polling logic (check every 5 minutes)  
* \[ \] Fetch new BuyItNow listings from eBay API  
* \[ \] Score with TF-IDF filter  
* \[ \] Store high-scoring listings (\>0.9 threshold)  
* \[ \] Send notifications (Discord webhook or email)  
* \[ \] Set up cron job on droplet

**Day 17: BuyItNow Frontend**

* \[ \] Create BuyItNowPage.tsx component  
* \[ \] Display recent alerts (last 24 hours)  
* \[ \] Add "Mark as Seen" button  
* \[ \] Add "Mark as Purchased" button  
* \[ \] Show listing details (price, seller, end time, score)  
* \[ \] Add countdown timer for listings  
* \[ \] Test with mock alerts

**Day 18: Tour View \- Structure**

* \[ \] Create TourView.tsx component  
* \[ \] Design tour navigation (prev/next arrows, progress dots)  
* \[ \] Create tour stops array:  
  * \[ \] Stop 1: Landing page (Record of the Day, MySpace aesthetic)  
  * \[ \] Stop 2: Dashboard (stats, recent annotations)  
  * \[ \] Stop 3: Discogs annotation view  
  * \[ \] Stop 4: eBay annotation view  
  * \[ \] Stop 5: Knapsack feature  
  * \[ \] Stop 6: Keepers inventory  
* \[ \] Add overlay/modal styling  
* \[ \] Implement keyboard navigation (arrow keys)

**Day 19: Tour View \- Content**

* \[ \] Write tour text for each stop (2-3 sentences)  
* \[ \] Add screenshots/visual highlights for each stop  
* \[ \] Add "Skip Tour" button  
* \[ \] Add "Start Tour" button on landing page  
* \[ \] Store tour completion in localStorage  
* \[ \] Test tour flow end-to-end

**Day 20: Documentation \- README**

* \[ \] Write project overview section  
* \[ \] Document tech stack (React, Go, Django, PostgreSQL, Redis)  
* \[ \] Write setup instructions (local development)  
* \[ \] Document API endpoints  
* \[ \] Add screenshots of key features  
* \[ \] Write deployment instructions (Digital Ocean)  
* \[ \] Add credits (Discogs, eBay, Claude APIs)

**Day 21: Documentation \- FAQ & Polish**

* \[ \] Write FAQ (10-15 common questions)  
  * \[ \] "What is Flic-a-Disc?"  
  * \[ \] "How does the recommendation system work?"  
  * \[ \] "What is the knapsack feature?"  
  * \[ \] "How accurate is the ML model?"  
  * \[ \] "Can I use this for my collection?"  
* \[ \] Add FAQ page to frontend  
* \[ \] Test all Week 3 features  
* \[ \] Fix bugs and polish UI

**Goal:** Public-facing features complete, documentation exists

---

### **Week 4: Testing Foundation**

**Day 22: Frontend Testing Setup**

* \[ \] Install Jest and React Testing Library  
* \[ \] Configure test environment (tsconfig, jest.config)  
* \[ \] Write first test (LandingPage renders)  
* \[ \] Set up test coverage reporting  
* \[ \] Add npm test script

**Day 23: Frontend Unit Tests \- Components**

* \[ \] Test AnnotationPage (Keep/Skip buttons work)  
* \[ \] Test KeepersPage (tabs, sorting, search)  
* \[ \] Test KnapsackPage (sliders, budget input)  
* \[ \] Test DashboardPage (stats display)  
* \[ \] Test TourView (navigation, stop content)  
* \[ \] Aim for 60%+ component coverage

**Day 24: Frontend Unit Tests \- Hooks & Utils**

* \[ \] Test usePaginate hook  
* \[ \] Test useAuth hook  
* \[ \] Test API client functions (apiFetch, mlFetch)  
* \[ \] Test utility functions (date formatting, price formatting)  
* \[ \] Aim for 80%+ hook/util coverage

**Day 25: Backend Testing Setup (Go)**

* \[ \] Set up testing structure (handlers\_test.go, etc.)  
* \[ \] Configure test database (in-memory SQLite or separate PG)  
* \[ \] Write first test (health check endpoint)  
* \[ \] Add make test command  
* \[ \] Set up test coverage reporting

**Day 26: Backend Unit Tests (Go)**

* \[ \] Test authentication handlers (login, logout, register)  
* \[ \] Test annotation handlers (submit batch, get records)  
* \[ \] Test keepers handlers (get, filter, search)  
* \[ \] Test knapsack handlers (optimize, get sellers)  
* \[ \] Aim for 60%+ handler coverage

**Day 27: TypeScript Strict Mode**

* \[ \] Enable strict mode in tsconfig.json  
* \[ \] Fix all type errors in src/ (one file at a time)  
* \[ \] Add proper types for all API responses  
* \[ \] Add proper types for all component props  
* \[ \] Verify no "any" types remain

**Day 28: Integration Tests & Buffer**

* \[ \] Write integration test: login → annotate → submit  
* \[ \] Write integration test: fetch keepers → filter → export  
* \[ \] Write integration test: knapsack optimize → purchase  
* \[ \] Fix any remaining bugs from testing  
* \[ \] Code cleanup and refactoring  
* \[ \] Review Month 1 progress

**Goal:** Testing infrastructure in place, critical paths covered

---

## **Month 2: Infrastructure \+ Optimization**

### **Week 5: Jenkins CI/CD (Part 1\)**

**Day 29: Jenkins Installation**

* \[ \] Install Docker on droplet  
* \[ \] Pull Jenkins Docker image  
* \[ \] Create docker-compose.yml for Jenkins  
* \[ \] Set up persistent volume for Jenkins data  
* \[ \] Start Jenkins container  
* \[ \] Complete initial setup wizard  
* \[ \] Install recommended plugins

**Day 30: Jenkins Configuration**

* \[ \] Configure Jenkins admin user  
* \[ \] Install additional plugins:  
  * \[ \] Git plugin  
  * \[ \] Docker plugin  
  * \[ \] Blue Ocean  
  * \[ \] Pipeline plugin  
* \[ \] Set up GitHub credentials in Jenkins  
* \[ \] Configure Docker credentials  
* \[ \] Test GitHub connection

**Day 31: Nginx Reverse Proxy**

* \[ \] Install certbot on droplet  
* \[ \] Generate SSL certificate for jenkins.yourdomain.com  
* \[ \] Configure nginx reverse proxy to Jenkins  
* \[ \] Set up HTTPS redirect  
* \[ \] Test Jenkins accessible via HTTPS  
* \[ \] Configure systemd for nginx auto-start

**Day 32: Basic Jenkinsfile**

* \[ \] Create Jenkinsfile in repo root  
* \[ \] Add stages: Test, Build, Deploy  
* \[ \] Write Test stage (placeholder for now)  
* \[ \] Write Build stage (placeholder)  
* \[ \] Write Deploy stage (placeholder)  
* \[ \] Push Jenkinsfile to GitHub  
* \[ \] Create Jenkins pipeline job pointing to repo

**Day 33: GitHub Webhooks**

* \[ \] Configure GitHub webhook to Jenkins  
* \[ \] Test webhook (push to repo triggers build)  
* \[ \] Configure branch filters (main, develop)  
* \[ \] Add build status badges to README  
* \[ \] Test end-to-end: commit → webhook → build

**Day 34: Jenkins Credentials & Secrets**

* \[ \] Add SSH key for droplet deployment  
* \[ \] Add Docker registry credentials  
* \[ \] Add environment variables (.env secrets)  
* \[ \] Test credential access in pipeline  
* \[ \] Document credential management

**Day 35: Buffer & Testing**

* \[ \] Test Jenkins restart (systemd service)  
* \[ \] Test Jenkins backup and restore  
* \[ \] Fix any Jenkins issues  
* \[ \] Document Jenkins setup steps  
* \[ \] Review Week 5 progress

**Goal:** Basic CI pipeline running

---

### **Week 6: Jenkins CI/CD (Part 2\)**

**Day 36: Parallel Testing Stage**

* \[ \] Update Jenkinsfile with parallel stages  
* \[ \] Add frontend test stage:  
  * \[ \] Install npm dependencies  
  * \[ \] Run npm test  
  * \[ \] Report test results  
* \[ \] Add backend test stage:  
  * \[ \] Run go test ./...  
  * \[ \] Report test results  
* \[ \] Add ML test stage:  
  * \[ \] Install Python dependencies  
  * \[ \] Run pytest  
  * \[ \] Report test results

**Day 37: Docker Image Building**

* \[ \] Write Dockerfile for React frontend (multi-stage)  
* \[ \] Write Dockerfile for Go backend  
* \[ \] Write Dockerfile for Django ML service  
* \[ \] Test local builds (docker build)  
* \[ \] Add Docker build stage to Jenkinsfile  
* \[ \] Tag images with git commit SHA

**Day 38: Container Registry**

* \[ \] Set up self-hosted Docker registry OR use Docker Hub  
* \[ \] Configure registry authentication  
* \[ \] Update Jenkinsfile to push images  
* \[ \] Test push to registry  
* \[ \] Implement image retention (keep last 10 versions)  
* \[ \] Add cleanup job for old images

**Day 39: Deployment Stage**

* \[ \] Write deployment script (deploy.sh)  
  * \[ \] SSH to droplet  
  * \[ \] Pull new images  
  * \[ \] Stop old containers  
  * \[ \] Start new containers  
  * \[ \] Run health checks  
* \[ \] Add deployment stage to Jenkinsfile  
* \[ \] Test deployment from Jenkins

**Day 40: Health Checks & Rollback**

* \[ \] Add health check endpoints:  
  * \[ \] /health on Go backend  
  * \[ \] /health on Django ML service  
  * \[ \] Root page load for React frontend  
* \[ \] Update deployment script to check health  
* \[ \] Add automatic rollback on health check failure  
* \[ \] Test rollback scenario

**Day 41: Dev \+ Prod Sync**

* \[ \] Create staging branch in Git  
* \[ \] Configure separate Jenkins job for staging  
* \[ \] Set up staging environment on droplet (separate ports)  
* \[ \] Deploy to staging automatically on develop branch  
* \[ \] Deploy to production manually (or on main branch merge)  
* \[ \] Document deployment workflow

**Day 42: Notifications & Polish**

* \[ \] Set up Discord webhook OR Slack integration  
* \[ \] Add build notifications (success, failure)  
* \[ \] Add deployment notifications  
* \[ \] Configure email notifications for failures  
* \[ \] Test all notification types  
* \[ \] Review Week 6 progress

**Goal:** Full CI/CD pipeline operational

---

### **Week 7: Monitoring \+ Observability**

**Day 43: Prometheus Setup**

* \[ \] Install Prometheus on droplet (Docker)  
* \[ \] Create prometheus.yml configuration  
* \[ \] Configure scrape targets:  
  * \[ \] Node exporter (system metrics)  
  * \[ \] Go backend /metrics endpoint  
  * \[ \] Django ML service /metrics endpoint  
* \[ \] Start Prometheus container  
* \[ \] Test Prometheus UI (localhost:9090)

**Day 44: Application Metrics (Go)**

* \[ \] Install prometheus Go client library  
* \[ \] Add /metrics endpoint to Go backend  
* \[ \] Expose metrics:  
  * \[ \] Request count (by endpoint, status code)  
  * \[ \] Request duration (histogram)  
  * \[ \] Active requests (gauge)  
  * \[ \] Error count  
* \[ \] Test metrics collection in Prometheus

**Day 45: Application Metrics (Django)**

* \[ \] Install django-prometheus  
* \[ \] Configure middleware  
* \[ \] Add /metrics endpoint  
* \[ \] Expose metrics:  
  * \[ \] Request count, duration, errors  
  * \[ \] ML prediction latency  
  * \[ \] Model accuracy (from recent batches)  
  * \[ \] Batch processing count  
* \[ \] Test metrics collection

**Day 46: Grafana Setup**

* \[ \] Install Grafana on droplet (Docker)  
* \[ \] Configure Grafana to use Prometheus data source  
* \[ \] Create first dashboard: System Health  
  * \[ \] CPU usage  
  * \[ \] Memory usage  
  * \[ \] Disk usage  
  * \[ \] Network I/O  
* \[ \] Test dashboard with live data

**Day 47: Application Dashboards**

* \[ \] Create dashboard: Go Backend  
  * \[ \] Request rate (requests/sec)  
  * \[ \] p50, p95, p99 latency  
  * \[ \] Error rate (%)  
  * \[ \] Endpoint breakdown  
* \[ \] Create dashboard: ML Service  
  * \[ \] Prediction latency  
  * \[ \] Batch accuracy over time  
  * \[ \] Model threshold over time  
  * \[ \] Online learning triggers

**Day 48: Alerting Rules**

* \[ \] Configure Alertmanager  
* \[ \] Write alert rules:  
  * \[ \] High error rate (\>5%)  
  * \[ \] High latency (p99 \>5s)  
  * \[ \] Low disk space (\<15%)  
  * \[ \] Service down (health check failing)  
* \[ \] Configure Discord webhook for alerts  
* \[ \] Test alerting (trigger intentional failures)

**Day 49: Logging & Cleanup**

* \[ \] Configure structured logging (JSON format)  
* \[ \] Set up log rotation  
* \[ \] Add request ID tracing through services  
* \[ \] Create logs dashboard in Grafana  
* \[ \] Document monitoring setup  
* \[ \] Review Week 7 progress

**Goal:** Complete observability stack

---

### **Week 8: ML Tooling**

**Day 50: MLflow Setup**

* \[ \] Install MLflow on droplet (pip install)  
* \[ \] Start MLflow server (mlflow server)  
* \[ \] Configure tracking URI in Django  
* \[ \] Create first experiment: Discogs Bandit  
* \[ \] Test logging a simple run

**Day 51: MLflow Integration \- Training**

* \[ \] Update trainer to log to MLflow:  
  * \[ \] Log hyperparameters (learning\_rate, batch\_size, threshold)  
  * \[ \] Log metrics (accuracy, F1, precision, recall)  
  * \[ \] Log model artifacts  
  * \[ \] Log training dataset info  
* \[ \] Test end-to-end: train model → view in MLflow UI

**Day 52: MLflow Integration \- Online Learning**

* \[ \] Update online learning to log each update:  
  * \[ \] Log batch accuracy  
  * \[ \] Log new triplets generated  
  * \[ \] Log threshold changes  
  * \[ \] Log model version  
* \[ \] Add tags for tracking (online\_learning, batch\_N)  
* \[ \] Test logging over 5 batches

**Day 53: Optuna Setup**

* \[ \] Install Optuna (pip install)  
* \[ \] Create hyperparameter optimization script  
* \[ \] Define parameter space:  
  * \[ \] learning\_rate: \[1e-5, 1e-2\]  
  * \[ \] batch\_size: \[16, 32, 64, 128\]  
  * \[ \] threshold: \[0.01, 0.5\]  
  * \[ \] hidden\_dims: \[\[128,64,32\], \[256,128,64\], \[512,256,128\]\]  
* \[ \] Run small optimization study (10 trials)

**Day 54: Optuna \+ MLflow Integration**

* \[ \] Integrate Optuna with MLflow (log all trials)  
* \[ \] Run full hyperparameter sweep (50 trials)  
* \[ \] Analyze results in MLflow UI  
* \[ \] Identify best hyperparameters  
* \[ \] Document optimal configuration

**Day 55: Model Retraining with Optimal Params**

* \[ \] Update trainer config with optimal hyperparameters  
* \[ \] Retrain model from scratch  
* \[ \] Evaluate on validation set  
* \[ \] Compare to previous model (accuracy, F1)  
* \[ \] Deploy new model if better  
* \[ \] Log deployment to MLflow

**Day 56: ML Monitoring Dashboard**

* \[ \] Create Grafana dashboard: ML Performance  
  * \[ \] Current model accuracy  
  * \[ \] Prediction distribution (histogram)  
  * \[ \] Threshold over time  
  * \[ \] Batch accuracy over last 100 batches  
* \[ \] Add MLflow link to dashboard  
* \[ \] Document ML tooling setup  
* \[ \] Review Month 2 progress

**Goal:** Production ML tooling in place, model optimized

---

## **Month 3: Advanced Features \+ Polish**

### **Week 9: React Optimization**

**Day 57: Compound Components \- Listing Cards**

* \[ \] Refactor listing card into compound components:  
  * \[ \] ListingCard (container)  
  * \[ \] ListingCard.Image  
  * \[ \] ListingCard.Title  
  * \[ \] ListingCard.Artist  
  * \[ \] ListingCard.Actions  
* \[ \] Test composition in AnnotationPage  
* \[ \] Update styling

**Day 58: Render Props \- Filtering Logic**

* \[ \] Create FilterProvider component with render props  
* \[ \] Extract filter logic (search, sort, conditions)  
* \[ \] Refactor KeepersPage to use FilterProvider  
* \[ \] Refactor AnnotationPage to use FilterProvider  
* \[ \] Test reusability

**Day 59: Custom Hooks Refactor**

* \[ \] Create useRecords hook (fetch, filter, sort)  
* \[ \] Create useKeepers hook  
* \[ \] Create useKnapsack hook  
* \[ \] Create useBuyItNow hook  
* \[ \] Replace inline logic with hooks across all pages  
* \[ \] Test hooks in isolation

**Day 60: Zustand State Management**

* \[ \] Install Zustand  
* \[ \] Create auth store (user, login, logout)  
* \[ \] Create annotation store (selected records, batch progress)  
* \[ \] Create keepers store (filters, sort)  
* \[ \] Replace prop drilling with Zustand  
* \[ \] Test state persistence

**Day 61: Virtual Scrolling Setup**

* \[ \] Install react-window  
* \[ \] Create VirtualList component wrapper  
* \[ \] Add to AnnotationPage (40 → 500 items visible)  
* \[ \] Add to KeepersPage (40 → 1000 items visible)  
* \[ \] Test performance (smooth scrolling)

**Day 62: Virtual Scrolling Optimization**

* \[ \] Optimize item rendering (React.memo)  
* \[ \] Add dynamic row height calculation  
* \[ \] Test with 10,000 item dataset  
* \[ \] Measure FPS improvement  
* \[ \] Document performance gains

**Day 63: React Polish & Buffer**

* \[ \] Fix any React warnings/errors  
* \[ \] Add React Suspense for code splitting  
* \[ \] Lazy load heavy components  
* \[ \] Test bundle size reduction  
* \[ \] Review Week 9 progress

**Goal:** React codebase modernized and performant

---

### **Week 10: Data Model Refinement**

**Day 64: Database Audit**

* \[ \] Review all tables and schemas  
* \[ \] Identify redundant fields  
* \[ \] Identify missing indexes  
* \[ \] Document normalization issues  
* \[ \] Create optimization plan

**Day 65: Records/Listings Table Cleanup**

* \[ \] Merge duplicate fields across Record/EbayListing  
* \[ \] Normalize artist/label tables (if not already)  
* \[ \] Add missing foreign keys  
* \[ \] Remove unused columns  
* \[ \] Write migration scripts

**Day 66: Index Optimization**

* \[ \] Analyze slow queries with EXPLAIN  
* \[ \] Add indexes on:  
  * \[ \] Foreign keys  
  * \[ \] Frequently filtered columns (artist, label, genre)  
  * \[ \] Timestamp columns (created\_at, updated\_at)  
* \[ \] Create composite indexes where needed  
* \[ \] Test query performance improvement

**Day 67: Auctions Table Design**

* \[ \] Design Auction model (listing\_id, current\_bid, end\_time, etc.)  
* \[ \] Design AuctionBid model (bid history)  
* \[ \] Create migrations  
* \[ \] Add auction monitoring logic (similar to BuyItNow)  
* \[ \] Test with mock auction data

**Day 68: Pricing Guidelines System**

* \[ \] Design SuggestedPrice table (record\_id, suggested\_price, confidence)  
* \[ \] Research pricing algorithms:  
  * \[ \] Recent sales average  
  * \[ \] Condition adjustments  
  * \[ \] Market trends  
* \[ \] Implement basic pricing model  
* \[ \] Test with known records

**Day 69: Data Migration & Testing**

* \[ \] Run all migrations on test database  
* \[ \] Verify data integrity  
* \[ \] Test application with new schema  
* \[ \] Fix any breaking changes  
* \[ \] Update API documentation

**Day 70: Database Backup Automation**

* \[ \] Write backup script (pg\_dump)  
* \[ \] Set up cron job (daily backups)  
* \[ \] Configure backup retention (keep last 30 days)  
* \[ \] Store backups off-droplet (Digital Ocean Spaces or S3)  
* \[ \] Test restore procedure  
* \[ \] Review Week 10 progress

**Goal:** Clean, optimized database schema

---

### **Week 11: Trading Platform Foundation**

**Day 71: Trading Platform Design Doc**

* \[ \] Write problem statement (multi-party record trading)  
* \[ \] Define user stories:  
  * \[ \] User creates wantlist with min\_take values  
  * \[ \] User lists inventory with max\_give values  
  * \[ \] User searches for trades  
  * \[ \] User proposes/accepts/declines trades  
* \[ \] Document constraints (balance, package limits, conditions)

**Day 72: Database Schema for Trading**

* \[ \] Design User model extensions (wants, haves, preferences)  
* \[ \] Design Want model (record\_id, min\_take, condition\_preference)  
* \[ \] Design Have model (record\_id, max\_give, condition)  
* \[ \] Design Trade model (participants, records, status)  
* \[ \] Design TradeProposal model (proposer, recipients, expiry)  
* \[ \] Create ER diagram

**Day 73: Cycle-Finding Algorithm**

* \[ \] Research NetworkX simple\_cycles()  
* \[ \] Design graph structure (users as nodes, potential trades as edges)  
* \[ \] Write pseudocode for:  
  * \[ \] Building trade graph  
  * \[ \] Finding all cycles up to length N  
  * \[ \] Verifying balance constraints  
  * \[ \] Ranking cycles by user preference  
* \[ \] Document algorithm complexity

**Day 74: Trade Graph Visualization Prototype**

* \[ \] Install graph visualization library (D3.js or Cytoscape.js)  
* \[ \] Create mock trade graph data  
* \[ \] Render users as nodes  
* \[ \] Render potential trades as edges  
* \[ \] Highlight cycles  
* \[ \] Test interactivity (click nodes/edges)

**Day 75: Trade Proposal UI Mockup**

* \[ \] Sketch trade proposal flow (Figma or paper)  
* \[ \] Design wantlist management UI  
* \[ \] Design inventory (haves) management UI  
* \[ \] Design search results UI (list of cycles)  
* \[ \] Design trade detail view (who gets what)  
* \[ \] Get feedback on mockups

**Day 76: Consignment Platform Design**

* \[ \] Write problem statement (sell records on user's behalf)  
* \[ \] Define consignment workflow:  
  * \[ \] User lists record for consignment  
  * \[ \] System suggests price  
  * \[ \] System lists on eBay/Discogs  
  * \[ \] System handles sale  
  * \[ \] System takes commission, pays user  
* \[ \] Document legal/trust considerations

**Day 77: Trading Platform Documentation**

* \[ \] Write technical spec (20+ pages):  
  * \[ \] Architecture overview  
  * \[ \] Database schema  
  * \[ \] API endpoints (planned)  
  * \[ \] Algorithms (cycle-finding, ranking)  
  * \[ \] UI mockups  
* \[ \] Review design with fresh eyes  
* \[ \] Identify Phase 1 vs Phase 2 features  
* \[ \] Review Week 11 progress

**Goal:** Clear design docs for future trading features

---

### **Week 12: Polish \+ Buffer**

**Day 78: Frontend Testing \- Comprehensive**

* \[ \] Write tests for all new Week 9 components  
* \[ \] Test virtual scrolling behavior  
* \[ \] Test Zustand stores  
* \[ \] Test custom hooks  
* \[ \] Aim for 80%+ coverage on new code

**Day 79: Backend Testing \- Comprehensive**

* \[ \] Write tests for new Week 10 endpoints  
* \[ \] Test database migrations (up and down)  
* \[ \] Test auction monitoring logic  
* \[ \] Test pricing algorithm  
* \[ \] Aim for 80%+ coverage on new code

**Day 80: End-to-End Testing**

* \[ \] Install Playwright  
* \[ \] Write E2E test: User signup → login  
* \[ \] Write E2E test: Annotate records → submit batch  
* \[ \] Write E2E test: Knapsack optimization → purchase  
* \[ \] Write E2E test: Filter keepers → export CSV  
* \[ \] Run all E2E tests in CI pipeline

**Day 81: Blue-Green Deployment**

* \[ \] Set up secondary environment (green)  
* \[ \] Update deployment script:  
  * \[ \] Deploy to green environment  
  * \[ \] Run health checks  
  * \[ \] Switch nginx proxy to green  
  * \[ \] Keep blue as backup  
* \[ \] Test deployment with zero downtime

**Day 82: Security & Vulnerability Scanning**

* \[ \] Install Trivy on CI pipeline  
* \[ \] Scan Docker images for vulnerabilities  
* \[ \] Fix critical/high vulnerabilities  
* \[ \] Add Trivy to automated builds  
* \[ \] Document security practices

**Day 83: Performance Optimization**

* \[ \] Review Grafana dashboards for bottlenecks  
* \[ \] Optimize slow database queries  
* \[ \] Add Redis caching for hot endpoints  
* \[ \] Optimize Docker image sizes (multi-stage builds)  
* \[ \] Measure improvement (before/after metrics)

**Day 84: Final Cleanup & Review**

* \[ \] Fix all remaining bugs from buffer list  
* \[ \] Code cleanup and refactoring  
* \[ \] Update all documentation  
* \[ \] Review entire 3-month sprint  
* \[ \] Celebrate progress\!

**Goal:** Production-ready system with solid testing

---

## **Summary**

### **Month 1 Deliverables (Days 1-28)**

* ✅ Knapsack purchasing system (with weight sliders, real-time optimization)  
* ✅ Complete keepers inventory (unified Discogs \+ eBay view, merge functionality)  
* ✅ eBay keepers annotation interface  
* ✅ BuyItNow automated monitoring and alerts  
* ✅ Interactive tour for visitors  
* ✅ README and FAQ documentation  
* ✅ Testing infrastructure (Jest, Go tests, 60-80% coverage)  
* ✅ TypeScript strict mode enabled

### **Month 2 Deliverables (Days 29-56)**

* ✅ Jenkins CI/CD pipeline (test, build, deploy)  
* ✅ GitHub webhooks and automated builds  
* ✅ Docker containerization and registry  
* ✅ Deployment automation with health checks and rollback  
* ✅ Dev \+ prod environment sync  
* ✅ Prometheus \+ Grafana monitoring stack  
* ✅ Application metrics and alerting (Discord/Slack)  
* ✅ MLflow experiment tracking  
* ✅ Optuna hyperparameter optimization  
* ✅ Optimized ML model deployed

### **Month 3 Deliverables (Days 57-84)**

* ✅ React modernization (compound components, render props, custom hooks, Zustand)  
* ✅ Virtual scrolling for 1000+ item lists  
* ✅ Optimized database schema (indexes, migrations, cleanup)  
* ✅ Auctions monitoring system  
* ✅ Pricing guidelines algorithm  
* ✅ Trading platform design documents (20+ pages)  
* ✅ Trade graph visualization prototype  
* ✅ Comprehensive testing (80%+ coverage, E2E with Playwright)  
* ✅ Blue-green deployment with zero downtime  
* ✅ Security scanning in CI (Trivy)

---

## **Daily Work Rhythm**

**Morning (2-3 hours):**

* Focus on the day's primary task (implementation, deep work)  
* No distractions, phone on silent

**Afternoon (1-2 hours):**

* Secondary tasks (testing, documentation, cleanup)  
* Code review and refactoring

**Evening (30-60 minutes, optional):**

* Learning (read docs, watch conference talks)  
* Plan next day's tasks  
* Update sprint checklist

**Weekly:**

* **Sunday evening:** Review week's progress, plan next week  
* **Saturday afternoon:** Buffer time for catching up or fixing bugs

---

## **Progress Tracking**

**Daily:**

* \[ \] Check off completed tasks in this document  
* \[ \] Note blockers or unexpected issues  
* \[ \] Update time estimates if tasks taking longer

**Weekly:**

* \[ \] Count completed vs planned tasks  
* \[ \] Adjust next week's plan if falling behind  
* \[ \] Celebrate wins (no matter how small)

**Monthly:**

* \[ \] Review milestone achievement (did you hit 70%+?)  
* \[ \] Document key learnings and breakthroughs  
* \[ \] Adjust remaining months' plans based on reality

---

## **Risky Assumptions (Why This Is Aggressive)**

1. **4-6 hours/day of focused work** \- Assumes minimal interruptions and high productivity  
2. **Minimal debugging spirals** \- Assumes most issues can be fixed within a day  
3. **Learning curve** \- Assumes picking up Jenkins, Prometheus, MLflow goes relatively smoothly  
4. **No scope creep** \- Assumes features don't expand during implementation  
5. **Infrastructure stability** \- Assumes droplet, databases, and APIs stay healthy

**Reality:** If you complete 70% of this plan, you'll have made massive progress. The aggressive timeline is designed to create momentum and push you forward, not to create stress.

---

## **Escape Valves (If Falling Behind)**

**Cut without major impact:**

* Trading platform design (defer to Month 4\)  
* Consignment platform design (not critical now)  
* Virtual scrolling (nice-to-have, can add later)  
* Auctions monitoring (BuyItNow is higher priority)  
* E2E testing with Playwright (integration tests cover most)

**Simplify without cutting:**

* MLflow: Track experiments manually in database instead  
* Optuna: Manually tune hyperparameters based on intuition  
* Blue-green deployment: Simple rolling deploys are fine  
* Comprehensive testing: Focus on critical paths (60% coverage OK)

**Defer to future sprints:**

* Pricing guidelines system (manual pricing OK for now)  
* Trade graph visualization (just design docs are enough)  
* Consignment features (much later)

---

## **Acceleration Opportunities (If Ahead)**

**Month 1 extras:**

* Add mobile-responsive styling  
* Implement dark mode toggle  
* Add video walkthrough to tour  
* Export keepers to CSV/JSON

**Month 2 extras:**

* Set up distributed tracing (Jaeger)  
* Implement A/B testing framework for models  
* Add canary deployments  
* Create staging environment for testing

**Month 3 extras:**

* Build API documentation (Swagger/OpenAPI)  
* Create admin dashboard for monitoring  
* Add user analytics tracking  
* Implement rate limiting per user

---

## **Final Thoughts**

**This plan is aggressive by design.** It's meant to motivate and create momentum, not to be followed rigidly.

**Key principles:**

* **Flexibility:** Adjust weekly based on reality  
* **Progress \> Perfection:** Done is better than perfect  
* **Learning:** Every challenge makes you better  
* **Momentum:** Small daily wins compound into massive progress

**By the end of 3 months, you will:**

* Have a production-ready, portfolio-worthy project  
* Know Jenkins, Prometheus, MLflow, and modern DevOps practices  
* Be significantly closer to 90% proficiency across your stack  
* Have shipped real features that you can demo and discuss  
* Feel like a much stronger, more complete engineer

**Remember:** 70% completion \= massive success. This plan pushes you forward, challenges you, and helps you grow. Adjust as needed, celebrate wins, and keep building.

**Good luck\! 🚀**

