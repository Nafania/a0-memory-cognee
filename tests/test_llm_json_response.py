import unittest

from helpers.llm_json import parse_llm_json_response


class LlmJsonResponseTest(unittest.TestCase):
    def test_merges_concatenated_arrays_without_duplicate_items(self):
        response = '["one", "two"]["one", "two"]'

        self.assertEqual(parse_llm_json_response(response), ["one", "two"])

    def test_merges_concatenated_arrays_preserving_new_items(self):
        response = '["one"]["two", "one"]'

        self.assertEqual(parse_llm_json_response(response), ["one", "two"])

    def test_keeps_identical_concatenated_objects(self):
        response = '{"action": "skip"}{"action": "skip"}'

        self.assertEqual(parse_llm_json_response(response), {"action": "skip"})

    def test_rejects_conflicting_concatenated_objects(self):
        response = '{"action": "skip"}{"action": "merge"}'

        with self.assertRaisesRegex(ValueError, "multiple JSON roots"):
            parse_llm_json_response(response)


if __name__ == "__main__":
    unittest.main()
