# Dockerfile
# Minimal image for running Claude Code CLI in agent eval containers.
FROM node:22-slim

# Install git (needed for repo operations) and ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /work
