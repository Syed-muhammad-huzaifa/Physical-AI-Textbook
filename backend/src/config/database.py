import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Centralized database configuration for Neon Postgres
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_6nN3gKuYvlCq@ep-crimson-shape-a43oif0g-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require",
)

# Use pool_pre_ping to keep connections healthy with Neon
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Yield a database session and ensure it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
