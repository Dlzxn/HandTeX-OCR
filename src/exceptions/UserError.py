


class ModelError(Exception):
    """
    Базовая ошибка модели
    """
    pass

class ModelNotFoundError(ModelError):
    """
    Ошибка нахождения модели по заданному пути
    """
    def __init__(self, model_name: str):
        super().__init__(f"Model {model_name} not found")

class ModelTrainingError(ModelError):
    """
    Ошибка обучения модели
    """
    def __init__(self, model_name: str, details: str):
        super().__init__(f"Model {model_name} training failed with error: {details}")