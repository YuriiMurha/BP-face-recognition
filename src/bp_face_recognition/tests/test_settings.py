import os
import unittest
from unittest.mock import patch
from bp_face_recognition.config.settings import Settings

class TestSettings(unittest.TestCase):
    def test_defaults(self):
        settings = Settings()
        self.assertEqual(settings.APP_NAME, "BP Face Recognition")
        self.assertFalse(settings.DEBUG)
        self.assertTrue(str(settings.DATA_DIR).endswith("data"))

    @patch.dict(os.environ, {"APP_NAME": "Test App", "DEBUG": "True", "DB_PORT": "6543"})
    def test_env_overrides(self):
        # We need to re-instantiate Settings to pick up env vars
        # Note: Pydantic settings might cache, but instantiating a new object usually works 
        # provided we don't rely on the global 'settings' instance from the module.
        settings = Settings() 
        self.assertEqual(settings.APP_NAME, "Test App")
        self.assertTrue(settings.DEBUG)
        self.assertEqual(settings.DB_PORT, 6543)

if __name__ == '__main__':
    unittest.main()
