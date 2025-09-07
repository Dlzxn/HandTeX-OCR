from abc import ABC, abstractmethod

class ContractSettings(ABC):
    @staticmethod
    @abstractmethod
    def get_device():
        pass

class CUActionContract(ABC):
    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def _save_model(self, name):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def update_model(self, model, name):
        pass


class DataInterface(ABC):
    @abstractmethod
    def load_data(self):
        pass

class TrainContract(ABC):
    @abstractmethod
    def start_train(self):
        pass


