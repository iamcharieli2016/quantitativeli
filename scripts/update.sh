#!/bin/bash

# Update script for Quantitative Trading LLM System

set -e

echo "🔄 Updating Quantitative Trading LLM System"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Pull latest changes
print_status "Pulling latest changes..."
git pull origin main

# Build and restart services
print_status "Building updated services..."
docker-compose build --no-cache

# Restart with new build
print_status "Restarting services with updates..."
docker-compose up -d --build

# Health check
print_status "Performing health checks..."
sleep 30

if docker-compose ps | grep -q "trading-app.*Up"; then
    print_status "Update completed successfully!"
else
    print_warning "Update completed but services may need attention"
fi

print_status "Use 'docker-compose logs -f trading-app' to check logs"