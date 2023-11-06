import json
import numpy as np
import os, sys
from collections import defaultdict
import itertools
import datetime
import pandas as pd
import json
import random
import argparse
from collections import Counter
import torch
pair2sequence = {
	2: tuple([(0 ,1)]),
	3: tuple([(0, 1), (0, 2), (1, 2)]),
	4: tuple([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
	5: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]),
	6: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5),
	    (4, 5)]),
	7: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5),
	    (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]),
	8: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3),
	    (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]),
	9: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
	    (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6),
	    (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]),
	10: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
	     (1, 7), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7),
	     (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9),
	     (7, 8), (7, 9), (8, 9)]),
	11: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
	     (1, 7), (1, 8), (1, 9), (1, 10), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 4), (3, 5),
	     (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 6), (5, 7), (5, 8),
	     (5, 9), (5, 10), (6, 7), (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)]),
	12: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
	     (1, 7), (1, 8), (1, 9), (1, 10), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11),
	     (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
	     (4, 10), (4, 11), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
	     (7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11)])
}

stat = {'MajorClaim-MajorClaim-Support': 0, 'MajorClaim-MajorClaim-Attack': 0, 'MajorClaim-MajorClaim-f': 0, 
	    'MajorClaim-Claim-Support': 0, 'MajorClaim-Claim-Attack': 0, 'MajorClaim-Claim-f': 0, 'Claim-MajorClaim-Support': 0, 'Claim-MajorClaim-Attack': 0, 'Claim-MajorClaim-f': 0,
	    'MajorClaim-Premise-Support': 0, 'MajorClaim-Premise-Attack': 0, 'MajorClaim-Premise-f': 0, 'Premise-MajorClaim-Support': 0, 'Premise-MajorClaim-Attack': 0, 'Premise-MajorClaim-f': 0,
	    'Claim-Claim-Support': 0, 'Claim-Claim-Attack': 0, 'Claim-Claim-f': 0,
	    'Claim-Premise-Support': 0, 'Claim-Premise-Attack': 0, 'Claim-Premise-f': 0, 'Premise-Claim-Support': 0, 'Premise-Claim-Attack': 0, 'Premise-Claim-f': 0,
	    'Premise-Premise-Support': 0,'Premise-Premise-Attack': 0, 'Premise-Premise-f': 0}


if __name__ == '__main__':
	data_df = pd.read_csv("pe_data_df.csv")
	AC_types_list = [list(eval(AC_types)) for AC_types in data_df["ac_types"]]
	AR_pairs_list = [eval(_) for _ in data_df['ac_rel_pairs']]
	AR_types_list = [eval(_) for _ in data_df['ac_rel_types']]
	for AC_types, AR_pairs, AR_types in zip(AC_types_list, AR_pairs_list, AR_types_list):
		
		span_num = len(AC_types)
		if span_num < 2:
			continue
		for pair in pair2sequence[span_num]:
			if pair in AR_pairs :
				id = AR_pairs.index(pair)
				stat[f'{AC_types[pair[0]]}-{AC_types[pair[1]]}-{AR_types[id]}'] += 1
			elif (pair[1], pair[0]) in AR_pairs:
				id = AR_pairs.index((pair[1], pair[0]))
				stat[f'{AC_types[pair[0]]}-{AC_types[pair[1]]}-{AR_types[id]}'] += 1
			else:
				stat[f'{AC_types[pair[0]]}-{AC_types[pair[1]]}-f'] += 1
	print(stat)
	
				