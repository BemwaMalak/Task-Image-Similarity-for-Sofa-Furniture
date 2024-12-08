from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .interface import DatabaseProvider

Base = declarative_base()


class SQLiteDatabaseProvider(DatabaseProvider):
    def __init__(self, database_url: str = "sqlite:///./products.db"):
        self.database_url = database_url
        self.engine = create_engine(
            database_url, connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._session = None

    def initialize_database(self) -> None:
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a database session"""
        if self._session is None:
            self._session = self.SessionLocal()
        return self._session

    def close_session(self) -> None:
        """Close the current database session"""
        if self._session is not None:
            self._session.close()
            self._session = None


# Singleton instance
_db_provider: Optional[DatabaseProvider] = None


def get_database_provider(
    database_url: str = "sqlite:///./products.db",
) -> DatabaseProvider:
    """Get the singleton instance of the database provider"""
    global _db_provider
    if _db_provider is None:
        _db_provider = SQLiteDatabaseProvider(database_url)
    return _db_provider
