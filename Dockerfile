# Dockerfile for bugeval-agent: runs Claude Code CLI for code review evaluation.
# Tools enabled: Read, Glob, Grep, Bash, WebSearch, WebFetch
# Network: full outbound (needed for WebSearch/WebFetch and the Anthropic API)
FROM node:22-slim

# System tools useful inside agent Bash sessions:
#   git       — repo operations
#   curl      — HTTP requests from Bash
#   ripgrep   — fast code search (rg)
#   jq        — JSON parsing from Bash
#   ca-certificates — TLS for outbound HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ripgrep \
    jq \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Run as non-root for defence-in-depth; /work is owned by this user.
RUN groupadd -r agent && useradd -r -g agent -d /work -s /bin/bash agent
RUN mkdir /work && chown agent:agent /work

USER agent
WORKDIR /work
