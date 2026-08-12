from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='NAVCOCKPIT_', env_file='.env', extra='ignore')

    host: str = '0.0.0.0'
    port: int = 8890

    cors_origins: list[str] = Field(default_factory=lambda: [
        'http://localhost:5173',
        'http://127.0.0.1:5173',
    ])

    mock_enabled: bool = True
    mock_tick_hz: float = 10.0
    mock_seed: int = 42

    ros2_enabled: bool = False
    ros2_namespace: str = ''


settings = Settings()
