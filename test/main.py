"""
Main file for running all tests


1. Importing the necessary modules
	First, you need to import the unittest module:

	`import unittest`

2. Define a test class
	Your test class should inherit from unittest.TestCase:

	```
	class MyTests(unittest.TestCase):
		pass
	```
3. Writing test methods
	Test methods in the test class should start with the word test. 
	This is how the test runner identifies test methods.

	```
	class MyTests(unittest.TestCase):
	def test_example(self):
		self.assertEqual(1 + 1, 2)
	```

4. Assertions
	unittest provides a variety of assertion methods to check for test outcomes:

	assertEqual(a, b): Check a and b are equal.
	assertNotEqual(a, b): Check a and b are not equal.
	assertTrue(x): Check that x is True.
	assertFalse(x): Check that x is False.
	assertIsNone(x): Check that x is None.
	assertIsNotNone(x): Check that x is not None.
	... and many more.

5. Running the tests
	There are several ways to run the tests:

	a. Using the Command Line:
	If your tests are in a file named test_example.py, you can run them using:

	```
	python -m unittest test_example.py
	```

	b. Using the unittest discover:

	If you have multiple test files, you can use the discover option:

	
	```
	python -m unittest discover
	# This will run all files named test*.py in the current directory.
	```

	c. Within the script:
	You can also add the following lines at the end of your test script 
	to run the tests when the script is executed:

	
	```
	if __name__ == '__main__':
		unittest.main()
	```

6. Test Setup and Teardown
	If you need to set up any state before tests run or clean up after they've run, 
	you can use the setUp and tearDown methods:

	```
	class MyTests(unittest.TestCase):
		def setUp(self):
			# Code to run before every test
			pass

		def tearDown(self):
			# Code to run after every test
			pass
			
	```

7. Test Class Setup and Teardown
	For setting up and tearing down at the class level 
	(rather than the individual test level), you can use setUpClass and 
	tearDownClass methods. These should be class methods:

```
	class MyTests(unittest.TestCase):
		@classmethod
		def setUpClass(cls):
			# Code to run before any tests in the class
			pass

		@classmethod
		def tearDownClass(cls):
			# Code to run after all tests in the class have run
        pass
```		

"""

# imports
import unittest

class INIT(unittest.TestCase):
	"""
	unittest provides a variety of assertion methods to check for test outcomes:

	assertEqual(a, b): Check a and b are equal.
	assertNotEqual(a, b): Check a and b are not equal.
	assertTrue(x): Check that x is True.
	assertFalse(x): Check that x is False.
	assertIsNone(x): Check that x is None.
	assertIsNotNone(x): Check that x is not None.
	"""
	def init(self):
		self.assertEqual(1+1, 2)
	

if __name__ == '__main__':
	unittest.main()