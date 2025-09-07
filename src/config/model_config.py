from pydantic import BaseModel


class PathConfig(BaseModel):
    train_path: str = "~/dev/python/HandTeX-OCR/src/data/archive"
    test_path: str = "src/data/archive"
    model_path: str = "models/model.pth"


class TrainConfig(BaseModel):
    name: str = "Math-To-TeX"
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 20
    shuffle: bool = True
    num_workers: int = 2


