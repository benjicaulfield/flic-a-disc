#!/bin/bash
# deploy.sh — deploy flic-a-disc to droplet

DROPLET_IP="$DOMAIN"
DROPLET_USER="root"
PROJECT_DIR="/opt/flic-a-disc"

echo "🚀 Starting deployment..."

# 1️⃣ Commit and push to GitHub
echo "📝 Committing changes..."
git add .
read -p "Commit message: " commit_msg
git commit -m "$commit_msg"
git push origin main

# 2️⃣ SSH into droplet and pull latest changes
echo "🔄 Pulling changes on droplet..."
ssh $DROPLET_USER@$DROPLET_IP << 'ENDSSH'
set -e
cd /opt/flic-a-disc

echo "📦 Fetching latest code..."
git fetch origin main
git reset --hard origin/main

# 3️⃣ Ensure Go and uv are installed
if ! command -v go &> /dev/null; then
    echo "Installing Go..."
    apt update && apt install -y golang
fi

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 4️⃣ Sync Python dependencies
cd python_services
uv sync --frozen

# 5️⃣ Rebuild Go backend
cd ../backend
echo "⚙️ Building Go backend..."
go build -o flic-go cmd/api/main.go

# 6️⃣ Restart services
echo "♻️ Restarting backend services..."
sudo systemctl restart flic-django
sudo systemctl restart flic-go
sudo systemctl restart nginx

# 7️⃣ Health checks
echo "🔍 Checking service health..."
sudo systemctl status flic-go --no-pager -l | head -10
sudo systemctl status flic-django --no-pager -l | head -10
sudo systemctl status nginx --no-pager -l | head -10

echo "✅ Deployment complete!"
ENDSSH

echo "🎉 Done! Check https://flic-a-disc.com"
