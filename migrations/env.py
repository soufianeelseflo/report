import os
import sys
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
    fileConfig(config.config_file_name)

# Add the 'app' directory to the Python path so models can be imported
# Adjust the path based on where alembic is run relative to the app code
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_dir)

# Import Base from your models file
# Ensure this path is correct relative to the project root
from autonomous_agency.app.db.models import Base
from autonomous_agency.app.core.config import settings # To get DB URL if needed

# Set the metadata target for 'autogenerate' support
target_metadata = Base.metadata

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
    # Alembic needs the synchronous URL format
    return f"postgresql://{user}:{password}@{server}:{port}/{db}"

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url() # Use the function to get URL from env vars
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include naming convention for autogenerate
        include_schemas=True, # If using schemas other than public
        compare_type=True, # Compare column types
        render_as_batch=True, # Recommended for SQLite, useful generally
        # Use the naming convention defined in models.py
        naming_convention=target_metadata.naming_convention,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Include naming convention for autogenerate
        compare_type=True,
        render_as_batch=True,
        # Use the naming convention defined in models.py
        naming_convention=target_metadata.naming_convention,
        )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Use the synchronous URL for engine creation within Alembic's online mode
    connectable = AsyncEngine(
        engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            # Use the synchronous URL here
            url=get_url(),
            future=True, # Required for SQLAlchemy 2.0 style
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())