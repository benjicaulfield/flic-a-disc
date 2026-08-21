# Build and Run Instructions

This project consists of three services: Go backend, Django Python services, and React frontend.

---

## Quick Start (Docker - Recommended)

### Prerequisites
- Docker Desktop installed
- At least 8GB RAM available
- ~10GB disk space

### Steps

1. **Pull Ollama model** (one-time setup):
   ```bash
   docker compose up ollama -d
   docker compose exec ollama ollama pull llama3.2:3b
   ```

2. **Start all services**:
   ```bash
   docker compose up
   ```

3. **Access the application**:
   - Frontend: http://localhost:5173
   - Go Backend: http://localhost:8080
   - Django API: http://localhost:8000
   - Django Admin: http://localhost:8000/admin

4. **Initial setup** (if needed):
   ```bash
   # Run migrations
   docker compose exec django python manage.py migrate

   # Ingest conversation files
   docker compose exec django python manage.py ingest_conversations

   # Create Django superuser (optional)
   docker compose exec django python manage.py createsuperuser
   ```

### Useful Docker Commands

```bash
# View logs
docker compose logs -f

# Restart a service
docker compose restart django

# Stop all services
docker compose down

# Stop and remove volumes (fresh start)
docker compose down -v

# Rebuild after code changes
docker compose up --build
```

---

## Manual Setup (Without Docker)

### Prerequisites
- Go 1.22+
- Python 3.11+
- Node.js 20+
- PostgreSQL 16+
- Ollama installed locally

### 1. Database Setup

```bash
# Install PostgreSQL
brew install postgresql@16

# Start PostgreSQL
brew services start postgresql@16

# Create database
createdb flicadisc
```

### 2. Ollama Setup

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull model (in another terminal)
ollama pull llama3.2:3b
```

### 3. Python Services (Django)

```bash
cd python_services

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy environment file
cp .env.example .env

# Edit .env with your database credentials
# DATABASE_URL=postgresql://user:password@localhost:5432/flicadisc

# Install dependencies
uv sync

# Run migrations
python manage.py migrate

# Ingest conversation files
python manage.py ingest_conversations

# Start Django server
python manage.py runserver 8000
```

### 4. Go Backend

```bash
cd backend

# Copy environment file
cp .env.example .env

# Edit .env if needed (defaults should work)

# Install dependencies
go mod download

# Run backend
go run cmd/main.go
```

### 5. React Frontend

```bash
cd frontend

# Copy environment file
cp .env.example .env.local

# Install dependencies
npm install

# Start dev server
npm run dev
```

### 6. Verify Setup

- Frontend: http://localhost:5173
- Go Backend: http://localhost:8080/health (should return status)
- Django API: http://localhost:8000/api/
- Test RAG: http://localhost:8080/api/rag/query/

---

## Development Workflow

### Hot Reloading

All services support hot reloading:
- **Django**: Auto-reloads on file changes
- **Go**: Use `air` for hot reload (optional)
- **React**: Vite dev server auto-reloads

### Running Tests

```bash
# Python tests
cd python_services
pytest

# Go tests
cd backend
go test ./...

# Frontend tests
cd frontend
npm test
```

### Database Migrations

```bash
# Create migration
cd python_services
python manage.py makemigrations

# Apply migration
python manage.py migrate
```

### Adding RAG Documents

```bash
# Manual ingestion
cd python_services
python manage.py ingest_conversations

# Or via API
curl -X POST http://localhost:8000/api/rag/ingest/ \
  -F "file=@/path/to/conversation.txt"
```

---

## Troubleshooting

### Port Conflicts

If ports are already in use, edit `docker-compose.yml` or `.env` files:
- Frontend: 5173
- Go Backend: 8080
- Django: 8000
- PostgreSQL: 5432
- Ollama: 11434

### Ollama Model Not Found

```bash
# Ensure model is pulled
ollama list

# Pull if missing
ollama pull llama3.2:3b
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
brew services list | grep postgresql

# Check connection
psql -h localhost -U flicuser -d flicadisc
```

### ChromaDB Persistence Issues

```bash
# Check data directory exists
mkdir -p python_services/rag_data/chroma_db

# Re-ingest if needed
python manage.py ingest_conversations --force
```

### Python Dependency Issues

```bash
# Clear and reinstall
cd python_services
rm -rf .venv uv.lock
uv sync
```

### Go Build Issues

```bash
# Clear cache and rebuild
cd backend
go clean -cache
go mod tidy
go build ./cmd
```

### Frontend Build Issues

```bash
# Clear and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

---

## Architecture Overview

```
┌──────────────┐
│   Browser    │ :5173
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Go Backend  │ :8080 (Proxy)
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│     Django       │ :8000
│  - REST API      │
│  - RAG endpoints │
└────┬─────────┬───┘
     │         │
     ▼         ▼
┌────────┐ ┌────────┐
│  Chroma│ │ Ollama │
│   DB   │ │  LLM   │
└────────┘ └────────┘
```

---

## Production Deployment

For production, consider:
- Use production Docker images (multi-stage builds)
- Set `DEBUG=False` in Django
- Use proper secrets management (not .env files)
- Set up reverse proxy (nginx)
- Use managed database (not local PostgreSQL)
- Configure CORS properly
- Set up monitoring and logging

---

## Resource Requirements

### Minimum
- 8GB RAM
- 4 CPU cores
- 10GB disk space

### Recommended
- 16GB RAM (for Llama 3.2 8B model)
- 6+ CPU cores
- 20GB disk space

### Models

- **Llama 3.2 3B**: ~2GB RAM, faster inference
- **Llama 3.2 8B**: ~5GB RAM, better quality

Choose based on available resources.

---

## Getting Help

If you encounter issues:
1. Check logs: `docker compose logs -f [service]`
2. Verify all services are healthy: `docker compose ps`
3. Check environment variables are set correctly
4. Ensure all prerequisites are installed
5. Review troubleshooting section above

For RAG-specific issues, check:
- Ollama is running: `curl http://localhost:11434/api/tags`
- ChromaDB has data: Check `python_services/rag_data/chroma_db/`
- Conversations were ingested: Django logs during startup
