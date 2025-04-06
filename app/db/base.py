import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings # We will create this next

# Construct the async database URL
DATABASE_URL = settings.ASYNC_DATABASE_URL

# Create the async engine
engine = create_async_engine(DATABASE_URL, echo=settings.DB_ECHO, future=True)

# Create a sessionmaker for async sessions
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False, # Important for async background tasks
    autoflush=False,
    autocommit=False
)

# Dependency to get DB session in FastAPI routes
async def get_db_session() -> AsyncSession:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit() # Commit changes made within the request lifecycle
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Function to get a session for background workers
async def get_worker_session() -> AsyncSession:
    # Workers manage their own commit/rollback/close cycles
    return AsyncSessionFactory()

# Import Base from models after defining engine and sessionmaker
from .models import Base