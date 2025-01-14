#!/usr/bin/env /bin/python3

import sys
import os

sample_int=1
sample_float=0.1
sample_complex=complex(0,1)
sample_bool=True
sample_string='A string'
sample_string_template="A string with " \
    f"a number within: {sample_int}"
sample_list=['a','b','c']
sample_tuple=tuple(['a', 'b', 'c'])
sample_dict={'a': 1, 'b':2, 'c': 3}
sample_set={'Simone', 'Chiesi'}
sample_frozenset=frozenset([range(0, 9, 1)])

def sample_function():
    """A sample function with its doctstring
    
    Returns:
        bool: Always True
    """
    try:
        if (1 in sample_frozenset):
            return True
        elif ('a' in sample_frozenset):
            return 2
        else:
            return False
    except Exception as e:
        return str(e)

class SampleClass:
    def __init__(self, a, b):
        self._a = a
        self._b = b
        pass

    def sampleMethod(self):
        suffix=''
        if (sample_function()):
            suffix=sample_string_template
        return "This is a sample output from" +\
            " SampleClass::sampleMethod! " + \
            suffix + f"a: {self._a}, b: {self._b}"

instance = SampleClass('a', 2)
print(instance.sampleMethod())
