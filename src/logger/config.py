import logging, time
import sys
from pathlib import Path

class TrainLogger:
    def __init__(self, log_file="logs/project.log"):
        self.logger = logging.getLogger("HandTeX-OCR")
        self.logger.setLevel(logging.DEBUG)

        Path("logs").mkdir(exist_ok=True)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)


    def log_in_epoch(self, epoch, loss):
        self.logger.info(f"Epoch {epoch} | Loss {loss.item():.4f}")


    def log_end_epoch(self, epoch, all_loss, loader, accuracy, time_start):
        self.logger.info(f"Epoch {epoch} | Avg Loss: {all_loss / len(loader):.4f} | "
                             f"Accuracy: {accuracy:.4f}")
        print(f"Прошло {time.time() - time_start} времени с начала обучения")


    def log_result(self, min_loss, max_accuracy):
        self.logger.info(f"Итог: минимальный loss модели: {min_loss}"
                         f"Максимальный accuracy модели: {max_accuracy}")