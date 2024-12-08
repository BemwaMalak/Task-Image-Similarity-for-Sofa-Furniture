from abc import ABC, abstractmethod

from sqlalchemy.orm import Session


class DatabaseProvider(ABC):
    @abstractmethod
    def initialize_database(self) -> None:
        """Initialize the database and create all tables"""
        pass

    @abstractmethod
    def get_session(self) -> Session:
        """Get a database session"""
        pass

    @abstractmethod
    def close_session(self) -> None:
        """Close the current database session"""
        pass
