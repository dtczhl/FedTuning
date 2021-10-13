"""
    Helper class to log data to file
"""


class FileLogger:

    def __init__(self, *, file_path: str):
        """ Write data to file
        :param file_path: path to the file
        """

        self.file_path = file_path

        try:
            self.file_writer = open(self.file_path, 'w')
        except IOError:
            print(f'Could not open {self.file_path} to write')
            exit(-1)

    def write(self, *, message: str) -> None:
        """ Write message to file
        :param message: message to log
        :return: None
        """

        try:
            self.file_writer.write(message)
            self.file_writer.flush()
        except IOError:
            print(f'Error write message {message}')
            exit(-1)

    def get_file_path(self) -> str:
        """ Return file path
        :return: file path
        """

        return self.file_path

    def close(self) -> None:
        """ Close file
        :return: None
        """

        try:
            self.file_writer.close()
        except IOError:
            print(f'Cannot close the file')
            exit(-1)
