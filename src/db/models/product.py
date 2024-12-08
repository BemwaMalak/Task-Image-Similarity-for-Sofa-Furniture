from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from ..provider import Base

class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    processed_file_path = Column(String)
    raw_file_path = Column(String)
    features_file_path = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Product {self.product_id}>"