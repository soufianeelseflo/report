# Stage 1: Base Image and System Dependencies
FROM python:3.11-slim AS base

# Set environment variables to prevent interactive prompts during installs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry specific (if you were using it)
    # POETRY_VERSION=1.7.1 \
    # POETRY_HOME="/opt/poetry" \
    # POETRY_NO_INTERACTION=1 \
    # PATH="$POETRY_HOME/bin:$PATH"
    # Node.js version
    NODE_VERSION=20

WORKDIR /app

# Install essential system packages + Node.js + Git
# Using NodeSource repository for up-to-date Node.js
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        git \
        build-essential \
        libpq-dev \
    # Add NodeSource repository and install Node.js
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_VERSION.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    # Clean up
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN node --version && npm --version

# Stage 2: Install Python Dependencies
FROM base AS python-deps

# Copy only requirements to leverage Docker cache
COPY ./autonomous_agency/requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Stage 3: Install Node.js Dependencies for open-deep-research
FROM python-deps AS node-deps

# Copy the open-deep-research repository code
# Assumes 'open-deep-research' directory is in the build context root
COPY ./open-deep-research /app/open-deep-research

# Install Node.js dependencies within the repo directory
WORKDIR /app/open-deep-research
# Use --legacy-peer-deps ONLY if necessary due to dependency conflicts in the repo
# RUN npm install --legacy-peer-deps --omit=dev --no-audit --no-fund --loglevel=error
RUN npm install --omit=dev --no-audit --no-fund --loglevel=error

# Switch back to the main app directory
WORKDIR /app

# Stage 4: Application Code & Final Image
FROM node-deps AS final

# Copy the rest of the application code
# Assumes 'autonomous_agency' directory is in the build context root
COPY ./autonomous_agency /app/autonomous_agency
COPY ./migrations /app/migrations
COPY ./alembic.ini /app/alembic.ini

# Copy the entrypoint script (we will create this next)
COPY ./docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint script to run migrations and then the app
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (passed to entrypoint script)
CMD ["uvicorn", "autonomous_agency.app.main:app", "--host", "0.0.0.0", "--port", "8000"]