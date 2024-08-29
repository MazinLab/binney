import unittest
from unittest import TestCase

class TestImportable(TestCase):
    def test_imports(self):
        import binney
        from binney import cli

if __name__ == "__main__":
    unittest.main()
