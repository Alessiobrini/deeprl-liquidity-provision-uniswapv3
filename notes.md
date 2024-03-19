# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:18:56 2024

@author: ab978
"""

# General Notes on Uniswap Env implementation

- Where to store variables for ex post analysis?
- Is the counter the best way to iterate? The code needs to be checked by chatGPT once is done.

## Step method

There is probably no need of coding the cash variable since the liquidity provision is active.
It starts with no allocation but since the first allocation is initalized, then it justa statys there
or change the interval width.