#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dgabot import DGABot
#import pytest 
import os
import pandas as pd
#import logging 

testfir = './tests/samples/' 

dgabot = DGABot() 


def test_init(): 
    """
    Basic test 
    """
    dgab = DGABot()
    assert 1 == 1 
    
