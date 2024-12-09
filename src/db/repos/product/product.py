from typing import List, Optional

from sqlalchemy.orm import Session

from ...models.product import Product
from .interface import IProductRepository


class ProductRepository(IProductRepository):
    def __init__(self, session: Session):
        self.session = session

    def create(self, product: Product) -> Product:
        self.session.add(product)
        self.session.commit()
        return product

    def get_by_id(self, product_id: str) -> Optional[Product]:
        return (
            self.session.query(Product).filter(Product.product_id == product_id).first()
        )

    def get_all(self) -> List[Product]:
        return self.session.query(Product).all()

    def update(self, product: Product) -> Product:
        existing_product = self.get_by_id(str(product.product_id))
        if existing_product:
            for key, value in product.__dict__.items():
                if not key.startswith("_"):
                    setattr(existing_product, key, value)
            self.session.commit()
        return existing_product

    def delete(self, product_id: str) -> bool:
        product = self.get_by_id(product_id)
        if product:
            self.session.delete(product)
            self.session.commit()
            return True
        return False

    def get_by_name(self, name: str) -> List[Product]:
        return self.session.query(Product).filter(Product.name.ilike(f"%{name}%")).all()
