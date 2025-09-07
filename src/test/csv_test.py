from src.utils.csv_check import CSVLogger

def test_csv():
    csv_log = CSVLogger(path = "../../models/test")
    csv_log.create_csv()
    csv_log.update_csv([2, 23, 14])
    csv_log.update_csv([3, 0.15, 10])
    result = csv_log.check_result()

    assert result == (1, 0)


if __name__ == '__main__':
    test_csv()