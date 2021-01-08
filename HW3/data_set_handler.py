class DataSetHandler:
    source_path: str

    def __init__(self, path: str):
        self.source_path = path

    def bulk_read(self, input_stream, bulk_size: int = 1):
        pass

    def read_all(self):
        pass
