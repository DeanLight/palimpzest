from palimpzest.elements import DataRecord
from palimpzest.elements import File

import os
import base64

class DataSource:
    """The base class for all data sources"""
    def __init__(self, basicElement):
        self.basicElement = basicElement

    def __str__(self):
        return f"{self.__class__.__name__}(basicElement={self.basicElement})"
    
    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""
    def __init__(self, path):
        super().__init__(File)
        self.path = path

    def __iter__(self):
        def filteredIterator():
            for x in os.listdir(self.path):
                file_path = os.path.join(self.path, x)
                if os.path.isfile(file_path):
                    dr = DataRecord(self.basicElement)
                    dr.filename = file_path
                    bytes_data = open(file_path, "rb").read()
                    dr.contents = base64.b64encode(bytes_data).decode('utf-8')

                    print("ABOUT TO YIELD DR", dr.filename, dr.contents[:10])
                    yield dr

        return filteredIterator()

class FileSource(DataSource):
    """FileSource returns a single File object from a single real-world local file"""
    def __init__(self, path):
        super().__init__(File)
        self.path = path

    def __iter__(self):
        def filteredIterator():
            dr = DataRecord(self.basicElement)
            dr.filename = self.path
            bytes_data = open(file_path, "rb").read()
            dr.contents = base64.b64encode(bytes_data).decode('utf-8')

            yield dr

        return filteredIterator()

#
# Other subclasses of DataSource could grab data from a database, a blob store, etc.
# The basicElement returned might not be a File, but instead a Record or Image or similar.
#