#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Wait for the database to be ready using pg_isready
echo "Waiting for database at $POSTGRES_SERVER:$POSTGRES_PORT..."
# Loop until pg_isready returns success (exit code 0)
until PGPASSWORD=$POSTGRES_PASSWORD pg_isready -h "$POSTGRES_SERVER" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done
>&2 echo "Postgres is up - executing command"

# Apply database migrations
echo "Applying database migrations..."
# Need to set PYTHONPATH so alembic can find the app code
export PYTHONPATH=/app
# Run alembic using the ini file at the root (copied to /app)
# Use the absolute path to alembic.ini within the container
alembic -c /app/alembic.ini upgrade head
echo "Database migrations applied."

# --- HARDENING START ---
# Check if any arguments were passed to the script
if [ $# -eq 0 ]; then
  echo "FATAL: No command arguments provided to docker-entrypoint.sh. Cannot execute." >&2
  exit 1
fi

# Optional: Check if the command looks like a faulty bash -c invocation
# This is a heuristic and might not catch all cases, the real fix is external config.
if [ "$1" = "bash" ] && [ "$2" = "-c" ] && [ -z "$3" ]; then
    echo "FATAL: Detected attempt to run 'bash -c' without a command argument." >&2
    echo "       Check container 'Command' override in deployment configuration (e.g., Coolify)." >&2
    exit 1
fi
# --- HARDENING END ---

# Execute the command passed as arguments (e.g., uvicorn ...)
echo "Executing command: $@"
exec "$@"

# Script will exit here upon successful exec, or earlier if set -e triggers or checks fail.
echo "Error: exec command failed." >&2
exit 1 # Explicitly exit with error if exec somehow fails