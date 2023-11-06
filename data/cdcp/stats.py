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
	2: tuple([(0, 1)]),
	3: tuple([(0, 1), (0, 2), (1, 2)]),
	4: tuple([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
	5: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]),
	6: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5),
		 (4, 5)]),
	7: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5),
		 (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]),
	8: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3),
		 (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7),
		 (6, 7)]),
	9: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
		 (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6),
		 (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]),
	10: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
		 (1, 7), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7),
		 (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9),
		 (7, 8), (7, 9), (8, 9)]),
	11: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (1, 2), (1, 3), (1, 4),
	           (1, 5),
	           (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
	           (3, 4),
	           (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 6),
	           (5, 7),
	           (5, 8), (5, 9), (5, 10), (6, 7), (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10),
	           (9, 10)]),

	12: tuple([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 2), (1, 3),
	           (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
	           (2, 8), (2, 9), (2, 10), (2, 11), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
	           (4, 5),
	           (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
	           (6, 7),
	           (6, 8), (6, 9), (6, 10), (6, 11), (7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (8, 10), (8, 11), (9, 10),
	           (9, 11), (10, 11)]),
	13: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 3), (2, 4), (2, 5),
		 (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
		 (3, 10), (3, 11), (3, 12), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (5, 6), (5, 7),
		 (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (7, 8), (7, 9),
		 (7, 10), (7, 11), (7, 12), (8, 9), (8, 10), (8, 11), (8, 12), (9, 10), (9, 11), (9, 12), (10, 11), (10, 12),
		 (11, 12)]),
	14: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (3, 4), (3, 5), (3, 6), (3, 7),
		 (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
		 (4, 12), (4, 13), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (6, 7), (6, 8), (6, 9),
		 (6, 10), (6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (8, 9), (8, 10),
		 (8, 11), (8, 12), (8, 13), (9, 10), (9, 11), (9, 12), (9, 13), (10, 11), (10, 12), (10, 13), (11, 12),
		 (11, 13), (12, 13)]),
	15: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
		 (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
		 (5, 13),
		 (5, 14), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (7, 8), (7, 9), (7, 10), (7, 11),
		 (7, 12), (7, 13), (7, 14), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (9, 10), (9, 11), (9, 12),
		 (9, 13), (9, 14), (10, 11), (10, 12), (10, 13), (10, 14), (11, 12), (11, 13), (11, 14), (12, 13), (12, 14),
		 (13, 14)]),
	16: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
		 (5, 12), (5, 13), (5, 14), (5, 15), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
		 (6, 15), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (8, 9), (8, 10), (8, 11),
		 (8, 12), (8, 13), (8, 14), (8, 15), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (10, 11), (10, 12),
		 (10, 13), (10, 14), (10, 15), (11, 12), (11, 13), (11, 14), (11, 15), (12, 13), (12, 14), (12, 15), (13, 14),
		 (13, 15), (14, 15)]),
	17: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
		 (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12),
		 (6, 13),
		 (6, 14), (6, 15), (6, 16), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16),
		 (8, 9),
		 (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14),
		 (9, 15),
		 (9, 16), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (11, 12), (11, 13), (11, 14), (11, 15),
		 (11, 16),
		 (12, 13), (12, 14), (12, 15), (12, 16), (13, 14), (13, 15), (13, 16), (14, 15), (14, 16), (15, 16)]),
	18: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9),
		 (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10),
		 (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
		 (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15),
		 (8, 16), (8, 17), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (10, 11), (10, 12),
		 (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17),
		 (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (13, 14), (13, 15), (13, 16), (13, 17), (14, 15), (14, 16),
		 (14, 17), (15, 16), (15, 17), (16, 17)]),
	19: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7),
		 (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8),
		 (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9),
		 (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10),
		 (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (8, 9), (8, 10), (8, 11), (8, 12),
		 (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15),
		 (9, 16), (9, 17), (9, 18), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18),
		 (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (12, 13), (12, 14), (12, 15), (12, 16),
		 (12, 17), (12, 18), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (14, 15), (14, 16), (14, 17), (14, 18),
		 (15, 16), (15, 17), (15, 18), (16, 17), (16, 18), (17, 18)]),
	20: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
		 (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
		 (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
		 (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
		 (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15),
		 (9, 16), (9, 17), (9, 18), (9, 19), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17),
		 (10, 18), (10, 19), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (12, 13),
		 (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18),
		 (13, 19), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (15, 16), (15, 17), (15, 18), (15, 19), (16, 17),
		 (16, 18), (16, 19), (17, 18), (17, 19), (18, 19)]),
	21: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
		 (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
		 (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
		 (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
		 (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14),
		 (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15),
		 (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17),
		 (11, 18), (11, 19), (11, 20), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20),
		 (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (14, 15), (14, 16), (14, 17), (14, 18),
		 (14, 19), (14, 20), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (16, 17), (16, 18), (16, 19), (16, 20),
		 (17, 18), (17, 19), (17, 20), (18, 19), (18, 20), (19, 20)]),
	22: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
		 (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
		 (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
		 (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
		 (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14),
		 (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13), (10, 14),
		 (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (11, 12), (11, 13), (11, 14), (11, 15),
		 (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17),
		 (12, 18), (12, 19), (12, 20), (12, 21), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20),
		 (13, 21), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (15, 16), (15, 17), (15, 18),
		 (15, 19), (15, 20), (15, 21), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (17, 18), (17, 19), (17, 20),
		 (17, 21), (18, 19), (18, 20), (18, 21), (19, 20), (19, 21), (20, 21)]),
	23: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9),
		 (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10),
		 (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11),
		 (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12),
		 (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13),
		 (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13), (10, 14),
		 (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (11, 12), (11, 13), (11, 14),
		 (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (12, 13), (12, 14), (12, 15),
		 (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (13, 14), (13, 15), (13, 16), (13, 17),
		 (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20),
		 (14, 21), (14, 22), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (16, 17), (16, 18),
		 (16, 19), (16, 20), (16, 21), (16, 22), (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (18, 19), (18, 20),
		 (18, 21), (18, 22), (19, 20), (19, 21), (19, 22), (20, 21), (20, 22), (21, 22)]),
	24: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9),
		 (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10),
		 (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11),
		 (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12),
		 (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13),
		 (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13),
		 (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (11, 12), (11, 13),
		 (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (12, 13),
		 (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (13, 14),
		 (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (13, 23), (14, 15), (14, 16),
		 (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 22), (14, 23), (15, 16), (15, 17), (15, 18), (15, 19),
		 (15, 20), (15, 21), (15, 22), (15, 23), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23),
		 (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23),
		 (19, 20), (19, 21), (19, 22), (19, 23), (20, 21), (20, 22), (20, 23), (21, 22), (21, 23), (22, 23)]),
	25: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9),
		 (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10),
		 (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11),
		 (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12),
		 (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13),
		 (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13),
		 (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (11, 12), (11, 13),
		 (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (12, 13),
		 (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (12, 24),
		 (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (13, 23), (13, 24),
		 (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 22), (14, 23), (14, 24), (15, 16),
		 (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), (16, 17), (16, 18), (16, 19),
		 (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23),
		 (17, 24), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), (19, 20), (19, 21), (19, 22), (19, 23),
		 (19, 24), (20, 21), (20, 22), (20, 23), (20, 24), (21, 22), (21, 23), (21, 24), (22, 23), (22, 24), (23, 24)]),
	26: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
		 (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
		 (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
		 (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
		 (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14),
		 (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13), (10, 14),
		 (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (11, 12), (11, 13), (11, 14),
		 (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (12, 13), (12, 14),
		 (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (12, 24), (13, 14),
		 (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (13, 23), (13, 24), (13, 25),
		 (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 22), (14, 23), (14, 24), (14, 25),
		 (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25), (16, 17),
		 (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (16, 25), (17, 18), (17, 19), (17, 20),
		 (17, 21), (17, 22), (17, 23), (17, 24), (17, 25), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24),
		 (18, 25), (19, 20), (19, 21), (19, 22), (19, 23), (19, 24), (19, 25), (20, 21), (20, 22), (20, 23), (20, 24),
		 (20, 25), (21, 22), (21, 23), (21, 24), (21, 25), (22, 23), (22, 24), (22, 25), (23, 24), (23, 25), (24, 25)]),
	27: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
		 (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
		 (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12),
		 (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13),
		 (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14),
		 (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13), (10, 14),
		 (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (11, 12), (11, 13), (11, 14),
		 (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (12, 13), (12, 14),
		 (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (12, 24), (13, 14),
		 (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (13, 23), (13, 24), (13, 25),
		 (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 22), (14, 23), (14, 24), (14, 25),
		 (14, 26), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25),
		 (15, 26), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (16, 25), (16, 26),
		 (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23), (17, 24), (17, 25), (17, 26), (18, 19), (18, 20),
		 (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26), (19, 20), (19, 21), (19, 22), (19, 23), (19, 24),
		 (19, 25), (19, 26), (20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (21, 22), (21, 23), (21, 24),
		 (21, 25), (21, 26), (22, 23), (22, 24), (22, 25), (22, 26), (23, 24), (23, 25), (23, 26), (24, 25), (24, 26),
		 (25, 26)]),
	28: tuple(
		[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (1, 2),
		 (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4),
		 (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 4), (3, 5), (3, 6),
		 (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (4, 5), (4, 6), (4, 7), (4, 8),
		 (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 9),
		 (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (6, 7), (6, 8), (6, 9), (6, 10),
		 (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (7, 8), (7, 9), (7, 10), (7, 11),
		 (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12),
		 (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (9, 10), (9, 11), (9, 12), (9, 13),
		 (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (10, 11), (10, 12), (10, 13), (10, 14),
		 (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (11, 12), (11, 13), (11, 14),
		 (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (11, 21), (11, 22), (11, 23), (12, 13), (12, 14),
		 (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (12, 24), (13, 14),
		 (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (13, 21), (13, 22), (13, 23), (13, 24), (13, 25),
		 (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (14, 21), (14, 22), (14, 23), (14, 24), (14, 25),
		 (14, 26), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (15, 23), (15, 24), (15, 25),
		 (15, 26), (15, 27), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (16, 25),
		 (16, 26), (16, 27), (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23), (17, 24), (17, 25), (17, 26),
		 (17, 27), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26), (18, 27), (19, 20),
		 (19, 21), (19, 22), (19, 23), (19, 24), (19, 25), (19, 26), (19, 27), (20, 21), (20, 22), (20, 23), (20, 24),
		 (20, 25), (20, 26), (20, 27), (21, 22), (21, 23), (21, 24), (21, 25), (21, 26), (21, 27), (22, 23), (22, 24),
		 (22, 25), (22, 26), (22, 27), (23, 24), (23, 25), (23, 26), (23, 27), (24, 25), (24, 26), (24, 27), (25, 26),
		 (25, 27), (26, 27)])
}

stat = {'value-value-reason': 0, 'value-value-evidence': 0, 'value-value-f': 0, 
	    'value-policy-reason': 0, 'value-policy-evidence': 0, 'value-policy-f': 0, 'policy-value-reason': 0, 'policy-value-evidence': 0, 'policy-value-f': 0,
	    'value-testimony-reason': 0, 'value-testimony-evidence': 0, 'value-testimony-f': 0, 'testimony-value-reason': 0, 'testimony-value-evidence': 0, 'testimony-value-f': 0,
		'value-fact-reason': 0, 'value-fact-evidence': 0, 'value-fact-f': 0, 'fact-value-reason': 0, 'fact-value-evidence': 0, 'fact-value-f': 0,
		'value-reference-reason': 0,'value-reference-evidence': 0, 'value-reference-f': 0, 'reference-value-reason': 0, 'reference-value-evidence': 0, 'reference-value-f': 0,

	    'policy-policy-reason': 0,'policy-policy-evidence': 0, 'policy-policy-f': 0,
	    'policy-testimony-reason': 0,'policy-testimony-evidence': 0, 'policy-testimony-f': 0, 'testimony-policy-reason': 0,'testimony-policy-evidence': 0, 'testimony-policy-f': 0,
	    'policy-fact-reason': 0,'policy-fact-evidence': 0, 'policy-fact-f': 0, 'fact-policy-reason': 0,'fact-policy-evidence': 0, 'fact-policy-f': 0,
	    'policy-reference-reason': 0,'policy-reference-evidence': 0, 'policy-reference-f': 0, 'reference-policy-reason': 0,'reference-policy-evidence': 0, 'reference-policy-f': 0,
	    
	    'testimony-testimony-reason': 0,'testimony-testimony-evidence': 0, 'testimony-testimony-f': 0,
		'testimony-fact-reason': 0,'testimony-fact-evidence': 0, 'testimony-fact-f': 0, 'fact-testimony-reason': 0,'fact-testimony-evidence': 0, 'fact-testimony-f': 0,
	    'testimony-reference-reason': 0, 'testimony-reference-evidence': 0,'testimony-reference-f': 0, 'reference-testimony-evidence': 0,'reference-testimony-reason': 0, 'reference-testimony-f': 0,
	    
		'fact-fact-reason': 0,'fact-fact-evidence': 0, 'fact-fact-f': 0,
		'fact-reference-reason': 0,'fact-reference-evidence': 0, 'fact-reference-f': 0, 'reference-fact-reason': 0,'reference-fact-evidence': 0, 'reference-fact-f': 0,
	    
		'reference-reference-reason': 0, 'reference-reference-evidence': 0,'reference-reference-f': 0,
		}

if __name__ == '__main__':
	data_df = pd.read_csv("cdcp_data_df2.csv")
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
	
				