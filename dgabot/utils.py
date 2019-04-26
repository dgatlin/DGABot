#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tldextract

def topLevel(df): 
    res = [] 
    for i in range(0,len(df)):
        t = tldextract.extract(df['Domain'][i])
        res.append(t.domain)
    df['Domain'] = res
    return df 


def validChars(X):
    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
    return valid_chars