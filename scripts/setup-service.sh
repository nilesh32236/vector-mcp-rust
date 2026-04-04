#!/bin/bash

# Vector MCP Rust Service Setup Script
# Run this as sudo if possible, or follow the manual steps below.

SERVICE_DIR="/etc/systemd/system"
PROJECT_DIR="/home/nilesh/Documents/vector-mcp/vector-mcp-rust"

echo "Configuring Vector MCP Rust service..."

# Copy service file to systemd
sudo cp "$PROJECT_DIR/scripts/vector-mcp-rust.service" "$SERVICE_DIR/"

# Reload systemd
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable vector-mcp-rust

# Start the service immediately
sudo systemctl start vector-mcp-rust

echo "Service has been configured and started."
echo "Check status:"
echo "systemctl status vector-mcp-rust"
echo "journalctl -u vector-mcp-rust -f"
