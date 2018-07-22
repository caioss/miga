import unittest

loader = unittest.TestLoader()
tests = loader.discover(".", "test_*.py")
runner = unittest.TextTestRunner(verbosity=1)
results = runner.run(tests)
