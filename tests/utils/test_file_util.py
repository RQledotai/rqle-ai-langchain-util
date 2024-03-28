from json import JSONDecodeError
from unittest import TestCase, main

from parameterized import parameterized

from rqle_ai_langchain_util.utils import file_util

class TestFileUtil(TestCase):

    @parameterized.expand([
        ['tests/resources', 'sample.txt', True],
        ['tests/resources', 'not_found.txt', False],
    ])
    def test_file_exists(self, directory, file_name, outcome):
        result = file_util.file_exists(directory, file_name)
        self.assertEqual(result, outcome)

    def test_read_file(self):
        result = file_util.read_file('tests/resources', 'sample.txt')
        self.assertEqual(result, 'This contains samples of the code.')

    @parameterized.expand([
        ['tests', 'resources', IsADirectoryError],
        ['tests/resources', 'not_found.txt', FileNotFoundError],
        ['tests/resources', 'corrupted_file.txt', OSError]
    ])
    def test_read_file_error(self, directory, file_name, error_type):
        with self.assertRaises(error_type):
            file_util.read_file(directory, file_name)

    def test_read_json_file(self):
        result = file_util.read_json_file('tests/resources', 'sample.json')
        self.assertEqual(result, {"key1": "value1"})

    @parameterized.expand([
        ['tests', 'resources', IsADirectoryError],
        ['test/resources', 'sample.txt', TypeError],
        ['tests/resources', 'not_found.json', FileNotFoundError],
        ['tests/resources', 'sample_error.json', JSONDecodeError]
    ])
    def test_read_json_file_error(self, directory, file_name, error_type):
        with self.assertRaises(error_type):
            file_util.read_json_file(directory, file_name)

    def test_read_image(self):
        result = file_util.read_image('img/rqle_ai_logo.jpg')
        self.assertTrue(result.startswith('/9j/4AAQSkZJRgABAQAAAQABAAD/4QAqRXhpZgAASUkqAAgAAAABADEBAgAHAAAAGgAAA'))

    @parameterized.expand([
        ['tests/resources', IsADirectoryError],
        ['tests/resources/not_found.jpg', FileNotFoundError]
    ])
    def test_read_image_error(self, file_path, error_type):
        with self.assertRaises(error_type):
            file_util.read_image(file_path)


if __name__ == '__main__':
    main()
