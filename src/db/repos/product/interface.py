from abc import ABC, abstractmethod
from typing import List, Optional

from ...models import Product


class IProductRepository(ABC):
    @abstractmethod
    def create(self, product: Product) -> Product:
        """Create a new product."""
        pass

    @abstractmethod
    def get_by_id(self, product_id: str) -> Optional[Product]:
        """Get a product by its ID."""
        pass

    @abstractmethod
    def get_all(self) -> List[Product]:
        """Get all products."""
        pass

    @abstractmethod
    def update(self, product: Product) -> Product:
        """Update an existing product."""
        pass

    @abstractmethod
    def delete(self, product_id: str) -> bool:
        """Delete a product by its ID."""
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> List[Product]:
        """Get products by name."""
        pass
