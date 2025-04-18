# Stage 1: Base Image and System Dependencies
FROM python:3.11-slim AS base

# Set environment variables to prevent interactive prompts during installs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Node.js version
    NODE_VERSION=20

WORKDIR /app

# Install essential system packages + Node.js + Git + netcat + postgresql-client (for pg_isready in entrypoint)
# Using NodeSource repository for up-to-date Node.js
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        git \
        build-essential \
        libpq-dev \
        netcat-traditional \
        postgresql-client \
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
RUN node --version && npm --version && psql --version

# Stage 2: Install Python Dependencies
FROM base AS python-deps

# Copy only requirements to leverage Docker cache
COPY ./Nexus Plan/requirements.txt . # UPDATED PATH
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

# Copy the rest of the application code and config files from the root
COPY ./Nexus Plan /app/Nexus Plan # UPDATED PATH
COPY ./migrations /app/migrations # Copy the migrations directory itself (contains versions/)
COPY ./alembic.ini /app/alembic.ini # Copy alembic config

# Create directory for generated website AND templates
RUN mkdir -p /app/static_website
RUN mkdir -p /app/Nexus Plan/app/templates # Create templates dir inside app structure

# Copy the UI template
COPY ./Nexus Plan/app/templates/control_panel.html /app/Nexus Plan/app/templates/

# Copy the pre-generated website files (if any, otherwise MCOL creates them)
COPY ./Nexus Plan/app/static_website/ /app/static_website/

# Copy the entrypoint script from the root
COPY ./docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint script to run migrations and then the app
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (passed to entrypoint script)
CMD ["uvicorn", "Nexus Plan.app.main:app", "--host", "0.0.0.0", "--port", "8000"] # UPDATED APP PATH