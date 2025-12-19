"""
Database models and connection for prediction logging
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL - use SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class PredictionLog(Base):
    """
    Table to log all predictions made by the model.
    Critical for drift detection and model monitoring.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Prediction results
    prediction = Column(Integer)  # 0 or 1
    probability_default = Column(Float)
    probability_no_default = Column(Float)

    # Model info
    model_version = Column(String, index=True)

    # Input features (stored as JSON for flexibility)
    features = Column(JSON)

    # Key features for drift monitoring (duplicated for quick access)
    loan_amount = Column(Float, index=True)
    property_value = Column(Float)
    income = Column(Float)
    credit_score = Column(Integer, index=True)
    credit_type = Column(String, index=True)
    loan_type = Column(String, index=True)
    ltv = Column(Float)
    dtir = Column(Float)
    region = Column(String, index=True)

    # Optional: Ground truth (if available later)
    actual_outcome = Column(Integer, nullable=True)  # To be updated when outcome is known

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, prediction={self.prediction}, timestamp={self.timestamp})>"


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_db():
    """
    Dependency for FastAPI to get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Create tables when run directly
    init_db()
    print("Database tables created!")
