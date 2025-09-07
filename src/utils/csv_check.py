import csv, os
from typing import Any

import numpy as np


class CSVLogger:
    def __init__(self, path: str):
        self.path = path

    def create_csv(self):
        with open(f"{self.path}.csv", "w", newline="") as file:
            pass

    def update_csv(self, data: list[str | int]) -> None:
        with open(f"{self.path}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([*data])

    def check_result(self) -> tuple[Any, Any]:
        with open(f"{self.path}.csv", "r", newline="") as file:
            reader = csv.reader(file)
            data = np.array(list(reader))
            return data[:, 1].argmin(), data[:, 2].argmax()







