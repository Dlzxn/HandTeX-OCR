import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, time

from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.datasets.MathData import Data
from src.logger.config import TrainLogger
from src.config.model_config import TrainConfig, PathConfig
from src.model.OCRmodel import Model
from src.exceptions.UserError import ModelTrainingError, ModelNotFoundError
from src.metrics.metrics import Metrics
from src.utils.csv_check import CSVLogger
from src.contracts.interface import (ContractSettings, CUActionContract,
                                     TrainContract, DataInterface)



class Settings(ContractSettings):
    @staticmethod
    def get_device() -> torch.device:
        """
        Поиск доступного девайся для обучения GPU/CPU
        :return: device.type
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CUModel(CUActionContract):
    """
    Class for CU actions with model
    C - create
    U - Update
    """
    def __init__(self):
        self.device_setting = Settings()
        self.path_cfg = PathConfig()
        self.model = Model()

    def _load_model(self) -> None:
        """
        Загрузка модели
        :return:
        """
        try:
            self._device = self.device_setting.get_device()
            self.model = Model()
            self.model.load_state_dict(torch.load(self.path_cfg.model_path, map_location=self._device))
            self.model.to(self._device)

        except FileNotFoundError:
            raise ModelNotFoundError("Math-to-TeX")

    def _create_model(self) -> None:
        """
        Создание модели
        :return:
        """
        self.model = Model()

    def _save_model(self, name: str | int) -> None:
        """
        Сохранение модели
        :return:
        """
        torch.save(self.model.state_dict(), self.path_cfg.model_path[:-4] + f"_{name}.csv")

    def get_model(self) -> Model:
        """
        Получение атрибута с моделью
        :return: model
        """
        try:
            self._load_model()
        except ModelNotFoundError:
            self._create_model()

        return self.model

    def update_model(self, model: Model, name: str | int = "") -> None:
        """
        Обновление и сохранение изменений модели
        :return: bool
        """
        self.model = model
        self._save_model(str(name))



class CreateData(DataInterface):
    def __init__(self):
        super().__init__()
        self.train_cfg = TrainConfig()
        self.path_cfg = PathConfig()

    def load_data(self) -> DataLoader:
        """
        Инициализация лоадера
        :return:
        """
        data = Data(self.path_cfg.train_path)
        dataset = data.get_dataset()
        return DataLoader(dataset, batch_size=self.train_cfg.batch_size,
                                 shuffle=self.train_cfg.shuffle)



class Train(TrainContract):
    """
    Основной класс для цикла обучения
    """
    def __init__(self):
        """
        Create CrossEntropyLoss + optimizer
        """
        super().__init__()

        self.actions = CUModel()
        self.train_cfg = TrainConfig()
        self.path_cfg = PathConfig()
        self.data = CreateData()
        self.logger = TrainLogger()
        self.writer = SummaryWriter("runs/model_exp")
        self.csv_logger = CSVLogger("models/data")

        self.metrics = Metrics(num_classes = 61,
                               writer = self.writer,
                               )
        self.loader = self.data.load_data()
        self.model = self.actions.get_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_cfg.learning_rate)

    def start_train(self) -> None:
        """
        Start training
        :return: None
        """
        self.csv_logger.create_csv()
        time_start = time.time()
        
        for epoch in range(self.train_cfg.epochs):
            self.model.train()
            all_loss: float = 0.0
            y_true: list[int] = []
            y_pred: list[int] = []

            for batch_num, (batch, labels) in enumerate(tqdm(self.loader,
                                                             desc=f"Epoch {epoch+1}/{self.train_cfg.epochs}")):
                self.optimizer.zero_grad()
                output = self.model(batch)

                preds = torch.argmax(output, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                loss = self.criterion(output, labels)
                all_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                if batch_num % 10 == 0:
                    # self.logger.log_in_epoch(epoch, loss)
                    self.writer.add_scalar("Loss/train", loss.item(),
                                           epoch * len(self.loader) + batch_num
                                           )

            self.model.eval()
            self.actions.update_model(self.model, name = epoch)

            accuracy = self.metrics.compute_accuracy(y_true, y_pred)
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)
            self.metrics.confusion_matrix(confusion_matrix(y_true, y_pred), classes=range(61))

            self.logger.log_end_epoch(epoch, all_loss, self.loader, accuracy, time_start)
            self.csv_logger.update_csv([epoch, all_loss / len(self.loader), accuracy])

        min_loss, max_accuracy = self.csv_logger.check_result()
        self.logger.log_result(min_loss, max_accuracy)

        self.writer.close()


