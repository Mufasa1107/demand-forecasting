from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    regions: tuple[str, ...] = ("North", "South", "East", "West")
    stores_per_region: int = 3
    categories_per_store: int = 3
    skus_per_category: int = 3
    periods: int = 180
    horizon: int = 14
    seed: int = 42
