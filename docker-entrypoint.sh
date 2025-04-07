#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Wait for the database to be ready using pg_isready
echo "Waiting for database at $POSTGRES_SERVER:$POSTGRES_PORT..."
until PGPASSWORD=$POSTGRES_PASSWORD pg_isready -h $POSTGRES_SERVER -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -q; do
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

# Execute the command passed as arguments (e.g., uvicorn ...)
echo "Executing command: $@"
exec "$@"