import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class InstallDependenciesTest(unittest.TestCase):
    def test_requirements_do_not_pin_litellm_below_cognee_requirement(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text()

        self.assertNotIn("litellm==", requirements)

    def test_install_allows_openai_two_for_cognee_litellm(self):
        hooks = (REPO_ROOT / "hooks.py").read_text()

        self.assertIn('pinned_openai = "openai<3"', hooks)
        self.assertNotIn("openai=={ver}", hooks)


if __name__ == "__main__":
    unittest.main()
