import os
import sys
import asyncio
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    # Use absolute path for fileConfig if running from different directories
    # Assumes alembic.ini is in the parent directory of script_location
    config_filepath = os.path.join(os.path.dirname(__file__), "..", "..", config.config_file_name)
    if os.path.exists(config_filepath):
        fileConfig(config_filepath)
    else:
        print(f"Warning: Alembic config file not found at expected location: {config_filepath}")


# Add the project root directory to the Python path so models can be imported
# Assumes this env.py is in autonomous_agency/migrations/
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_dir)

# Import Base from your models file
try:
    # Ensure the app package is importable
    from autonomous_agency.app.db.models import Base # Corrected import path
    target_metadata = Base.metadata
except ImportError as e:
    print(f"Error importing Base model: {e}")
    print("Ensure your project structure and PYTHONPATH are correct.")
    target_metadata = None


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def get_url():
    """Gets the database URL from environment variables."""
    user = os.getenv("POSTGRES_USER", "agencyuser")
    password = os.getenv("POSTGRES_PASSWORD", "supersecretpassword")
    server = os.getenv("POSTGRES_SERVER", "db")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "agencydb")
    # Alembic needs the synchronous URL format for config, but engine uses asyncpg
    return f"postgresql://{user}:{password}@{server}:{port}/{db}"

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    if target_metadata is None:
        print("Target metadata not loaded. Cannot run offline migrations.")
        return
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        compare_type=True,
        render_as_batch=True,
        naming_convention=target_metadata.naming_convention if hasattr(target_metadata, 'naming_convention') else None,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Helper function to run migrations within a transaction."""
    if target_metadata is None:
        print("Target metadata not loaded. Cannot run online migrations.")
        return
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=True,
        naming_convention=target_metadata.naming_convention if hasattr(target_metadata, 'naming_convention') else None,
        )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Use the synchronous URL from get_url() for engine_from_config
    connectable_config = config.get_section(config.config_ini_section)
    connectable_config['sqlalchemy.url'] = get_url() # Ensure sync URL is used for config

    connectable = AsyncEngine(
        engine_from_config(
            connectable_config,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        # Pass the synchronous connection to do_run_migrations
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # Load environment variables if running locally for online mode
    # In Docker, entrypoint handles this
    if os.getenv("ALEMBIC_LOCAL_RUN"): # Add a flag if needed for local online runs
        from dotenv import load_dotenv
        env_path = os.path.join(project_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
        else:
            print(f"Warning: .env file not found at {env_path} for local Alembic run.")
    asyncio.run(run_migrations_online())