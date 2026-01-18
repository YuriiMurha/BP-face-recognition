from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Base Paths
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    ASSETS_DIR: Path = ROOT_DIR / "assets"
    SRC_DIR: Path = ROOT_DIR / "src"

    # Data Subdirectories
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    LOGS_DIR: Path = DATA_DIR / "logs"
    
    # Dataset Specifics
    AUGMENTED_DIR: Path = DATASETS_DIR / "augmented"
    CROPPED_DIR: Path = DATASETS_DIR / "cropped"

    # Models
    MODELS_DIR: Path = SRC_DIR / "models"
    
    # App Settings
    APP_NAME: str = "BP Face Recognition"
    DEBUG: bool = False
    
    # Database (Example - could be loaded from .env)
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "user"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "faces_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
