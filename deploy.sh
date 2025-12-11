#!/bin/bash
# deploy.sh - Deploy flic-a-disc to droplet

DROPLET_IP="$DOMAIN"
DROPLET_USER="root"
PROJECT_DIR="/opt/flic-a-disc"

echo "🚀 Starting deployment..."

# 1. Commit and push to GitHub
echo "📝 Committing changes..."
git add .
read -p "Commit message: " commit_msg
git commit -m "$commit_msg"
git push origin main

# 2. SSH into droplet and pull changes
echo "🔄 Pulling changes on droplet..."
ssh $DROPLET_USER@$DROPLET_IP << 'ENDSSH'
set -e
cd /opt/flic-a-disc

echo "📦 Fetching latest code..."
git fetch origin main
git checkout main
git reset --hard origin/main

# 3. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.cargo/bin:\$PATH"
fi

# 4. Sync Python deps
cd python_services
uv sync

# 4.5 Rebuild Go backend
cd ../backend
go build -o flic-go ./cmd/server

# 5. Restart services
echo "♻️ Restarting services..."
sudo systemctl restart flic-django
sudo systemctl restart flic-go
sudo systemctl restart nginx

# 6. Check status
echo "✅ Service status:"
sudo systemctl status flic-django --no-pager -l | head -10
sudo systemctl status flic-go --no-pager -l | head -10
ENDSSH

echo "🎉 Deployment complete! Visit https://flic-a-disc.com"
