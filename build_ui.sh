#!/bin/bash

# Exit on error
set -e

echo "🚀 Building Local Code Wiki UI..."

# Navigate to frontend
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  npm install
fi

# Build and export
echo "🏗️  Building Next.js app..."
npm run build

# Clear existing public directory in root
echo "🧹 Cleaning up public directory..."
cd ..
mkdir -p public
rm -rf public/* || true

# Copy exported files to public
echo "🚚 Deploying to public directory..."
cp -r frontend/out/. public/

echo "✅ UI Build Complete!"
