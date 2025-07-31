#!/bin/bash

# Quantitative Trading LLM System Deployment Script

set -e

echo "🚀 Starting Quantitative Trading LLM System Deployment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="quant-trading-llm"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed"
}

# Create backup
create_backup() {
    if [ -d "data" ]; then
        print_status "Creating backup of existing data..."
        mkdir -p "$BACKUP_DIR"
        cp -r data "$BACKUP_DIR/"
        cp -r logs "$BACKUP_DIR/" 2>/dev/null || true
        print_status "Backup created at $BACKUP_DIR"
    fi
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Created .env file from .env.example. Please update it with your actual values."
    else
        print_status ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data logs config backups monitoring/rules
    chmod 755 data logs config
}

# Check required environment variables
check_env_vars() {
    print_status "Checking required environment variables..."
    
    required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "OPENAI_API_KEY"
        "SECRET_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            print_warning "Environment variable $var is not set"
        fi
    done
}

# Build and start services
deploy() {
    print_status "Building and starting services..."
    
    # Build the application
    docker-compose build --no-cache
    
    # Start services
    docker-compose up -d postgres redis
    
    # Wait for databases to be ready
    print_status "Waiting for databases to be ready..."
    sleep 30
    
    # Start the application
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 60
}

# Health check
health_check() {
    print_status "Performing health checks..."
    
    # Check if services are running
    services=("postgres" "redis" "trading-app" "prometheus" "grafana")
    
    for service in "${services[@]}"; do
        if docker-compose ps | grep -q "$service.*Up"; then
            print_status "$service is running"
        else
            print_error "$service is not running"
            return 1
        fi
    done
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    # Wait for PostgreSQL
    until docker-compose exec postgres pg_isready -U quant_user -d quant_trading; do
        print_status "Waiting for PostgreSQL..."
        sleep 5
    done
    
    # Run database initialization
    docker-compose exec postgres psql -U quant_user -d quant_trading -f /docker-entrypoint-initdb.d/init.sql
}

# Show service URLs
show_urls() {
    print_status "Deployment completed successfully!"
    echo ""
    echo "📊 Service URLs:"
    echo "- Main Application: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/docs"
    echo "- Prometheus: http://localhost:9090"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
    echo "- PostgreSQL: localhost:5432"
    echo "- Redis: localhost:6379"
    echo ""
    echo "📋 Useful commands:"
    echo "- View logs: docker-compose logs -f trading-app"
    echo "- Stop services: docker-compose down"
    echo "- Restart services: docker-compose restart"
    echo "- Update: ./scripts/update.sh"
}

# Main deployment flow
main() {
    print_status "Starting deployment process..."
    
    check_docker
    create_backup
    create_directories
    setup_env
    check_env_vars
    deploy
    health_check
    init_database
    show_urls
}

# Handle script arguments
case "${1:-}" in
    "check")
        check_docker
        check_env_vars
        ;;
    "backup")
        create_backup
        ;;
    "logs")
        docker-compose logs -f trading-app
        ;;
    "stop")
        docker-compose down
        ;;
    "restart")
        docker-compose restart
        ;;
    "update")
        docker-compose pull
        docker-compose up -d --build
        ;;
    *)
        main
        ;;
esac