import unittest
from unittest.mock import Mock
from pathlib import Path
from sasctl import pzmm

class TestScoreWrapper(unittest.TestCase):
    def test_write_score_wrapper_function_input(self):
        # mock imports
        imports = ['torch', 'pandas']
        score_function_definition = 'def score_function():'
        score_function_body = 'print("Scoring")'
        score_function_exception_body = 'print("Error in scoring")'
        items_to_return = 'result'
        model_load = 'model = pickle.load(f)'
        model_name_with_file_extension = 'model.pkl'
        output_variables = ['output_1', 'output_2']

        # mok the write_score_wrapper_function_input method
        result = pzmm.ScoreWrapper.write_score_wrapper_function_input(imports, score_function_definition, score_function_body, score_function_exception_body, items_to_return, model_load, model_name_with_file_extension, output_variables)

        # assert
        self.assertIn('import torch', result)
        self.assertIn('import pandas', result)
        self.assertIn('def score_function():', result)
        self.assertIn('print("Scoring")', result)
        self.assertIn('result', result)
        self.assertIn('model = pickle.load(f)', result)
        self.assertIn('model.pkl', result)
        self.assertIn('output_1, output_2', result)

    def test_validate_score_wrapper_syntax(self):
        # mock code
        code = 'def score_function():\n\tprint("Scoring")\n\tresult = model.predict(data)\n\treturn result\n'
        #send through validate score_wrapper_syntax
        result = pzmm.ScoreWrapper.validate_score_wrapper_syntax(code)
        #assert
        self.assertTrue(result)

