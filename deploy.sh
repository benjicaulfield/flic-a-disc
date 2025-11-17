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
cd /opt/flic-a-disc
git reset --hard origin/main
git pull origin main

# 3. Install any new Python dependencies
cd python_services
source venv/bin/activate
pip install -r requirements.txt --quiet

# 4. Restart services
echo "♻️  Restarting services..."
sudo systemctl restart flic-django
sudo systemctl restart flic-go
sudo systemctl restart nginx

# 5. Check status
echo "✅ Service status:"
sudo systemctl status flic-django --no-pager -l | head -5
sudo systemctl status flic-go --no-pager -l | head -5

echo "✅ Deployment complete!"
ENDSSH

echo "🎉 Done! Check https://flic-a-disc.com"