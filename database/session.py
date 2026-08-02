"""
Database engine and session factory. Everything else in the app imports
`SessionLocal` from here rather than creating its own connection — one
place to configure connection pooling, echo settings, etc.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    """Yield a session, ensuring it's closed even if an error occurs.
    Usage:
        with get_session() as session:
            ...
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()