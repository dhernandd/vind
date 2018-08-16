'''
Collection of little pythonic tools. Might need to organize this better in the future. 

@author: danielhernandez
'''

import datetime
import string


def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16]
    return s + '_D' + date




if __name__ == "__main__":
    print(addDateTime('Hello'))
    print(addDateTime())