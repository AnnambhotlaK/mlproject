import sys
from src.logger import logging

"""
Create custom error message.
"""
def error_message_detail(error, error_detail: sys):
    # exc_info gives info on error location and reason
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    error_message = "Error occurred in python script name [{0}] line [{1}] error message [{2}]".format(
        file_name, line_no, str(error)
    )

    return error_message


"""
CustomException handles exceptions from error_message_detail.
"""
class CustomException(Exception):
    
    """
    Initializes a new CustomException.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    """
    Returns CustomException as a String.
    """
    def __str__(self):
        return self.error_message