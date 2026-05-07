import unittest

from helpers.llm_json import format_llm_json_for_log, parse_llm_json_response


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

    def test_formats_parsed_json_for_logs_without_concatenated_roots(self):
        response = '["one", "two"]["one", "two"]'
        parsed = parse_llm_json_response(response)

        formatted = format_llm_json_for_log(parsed)

        self.assertEqual(formatted, '[\n  "one",\n  "two"\n]')
        self.assertNotIn("][", formatted)


if __name__ == "__main__":
    unittest.main()
