from enum import Enum

from src.model.house_price_predictor import HousePricePredictor


class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]

    @classmethod
    def validation(cls, name: str):
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")


class Models(CustomEnum):
    HOUSE_PRICE_PREDICTOR = HousePricePredictor  # python main.py train --model_name movie_predictor
