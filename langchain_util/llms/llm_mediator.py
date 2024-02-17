import json
import os
from pathlib import Path


def file_exists(file_dir: str, file_name: str) -> bool:
    """
    :param file_dir: Directory in which JSON file is stored
    :param file_name: Name of the file
    :return: True if the file exists, False otherwise
    """
    return os.path.exists(os.path.join(file_dir, file_name))


def read_file(file_dir: str, file_name: str) -> str:
    """
    :param file_dir: Directory in which JSON file is stored
    :param file_name: Name of the file
    :return: String representation of the content in the file
    """
    try:
        input_file = Path(os.path.join(file_dir, file_name))

        # test whether the file is a directory
        if input_file.is_dir():
            raise IsADirectoryError(f'{input_file} is not a file')

        return input_file.read_text(encoding='utf-8')

    except FileNotFoundError:
        raise FileNotFoundError(f'{file_name} not found in {file_dir}')
    except Exception as e:
        raise IOError(f'Error reading {file_name}: {e}')


def read_json_file(file_dir: str, file_name: str) -> dict:
    """
    :param file_dir: Directory in which JSON file is stored
    :param file_name: Name of the JSON file
    :return: JSON  representation of the content in the JSON file
    """
    # test whether the file has the extension .json
    if not file_name.endswith('.json'):
        raise TypeError(f'{file_name} is not a JSON file')

    try:
        input_file = Path(os.path.join(file_dir, file_name))

        # test whether the file is a directory
        if input_file.is_dir():
            raise IsADirectoryError(f'{input_file} is not a JSON file')

        with input_file.open('r') as f:
            data = json.load(f)

        return data
    except FileNotFoundError:
        raise FileNotFoundError(f'{file_name} not found in {file_dir}')
    except json.JSONDecodeError as jde:
        raise json.JSONDecodeError(msg=f'{file_name} is not a valid JSON file', doc=jde.doc, pos=jde.pos)
    except Exception as e:
        raise IOError(f'Error reading {file_name}: {e}')