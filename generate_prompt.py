import os
import ujson as json
import zipfile
import numpy as np
import pickle
import torch
import scipy.sparse as sp
import torch.nn as nn
from collections import defaultdict
from transformers import BertTokenizer
from models.pos_map import pair2sequence, bert_prefix_ac_map, pair_idx_map
from sklearn.metrics import f1_score
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig, LongformerForMaskedLM, \
    BartTokenizerFast, BartTokenizer, AutoTokenizer
from utils.basic_utils import get_edge_frompairs


# 2 1 [0]
# 3 3 [0, 1, 9]
# 4 6 [0, 1, 2, 9, 10, 18]
# 5 10 [0, 1, 2, 3, 9, 10, 11, 18, 19, 27]
# 6 15 [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35]
# 7 21 [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42]
# 8 28 [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48]
# 9 36 [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36, 37, 38, 42, 43, 44, 48, 49, 53]
# 10 45 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57]
# 11 54 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60]
# 12 63 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

# bert_weights_path = "./data/bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(bert_weights_path)
# vocab_size = 30522

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# plm_weights_path = "./data/longformer-base"
plm_weights_path = "./data/bart-base"

# special_tokens = ['<pad>', '<essay>', '<para-conclusion>', '<para-body>', '<para-intro>', '<ac>',
#                                '</essay>', '</para-conclusion>', '</para-body>', '</para-intro>', '</ac>']
# special_tokens_dict = {'additional_special_tokens': special_tokens}
# tokenizer = LongformerTokenizer.from_pretrained(plm_weights_path)

# tokenizer.add_special_tokens(special_tokens_dict)  ############为 roberta 设置特殊字符
# longformerconfig = LongformerConfig.from_pretrained(plm_weights_path)
# longformerconfig.attention_mode = 'sliding_chunks'
# longformerconfig.attention_window = [768] * 12

# longformer = LongformerForMaskedLM.from_pretrained(plm_weights_path, config=longformerconfig)
# longformer.resize_token_embeddings(len(tokenizer))
# vocab_size = longformer.vocab_size
# print("vocab_size", vocab_size)


special_tokens = ['<essay>', '<para-conclusion>', '<para-body>', '<para-intro>', '<ac>',
                               '</essay>', '</para-conclusion>', '</para-body>', '</para-intro>', '</ac>']
tokenizer = AutoTokenizer.from_pretrained(plm_weights_path, add_special_tokens=True)
tokenizer.add_tokens(special_tokens)  ############为 bart 设置特殊字符
vocab_size = len(tokenizer)
print("vocab_size", vocab_size)

# label_dict = {0: 'MajorClaim', 1: 'Claim', 2: 'Premise', 3: 'no relation', 4: 'relation', 5: "Support", 6: "Attack"}
# # tokenizer = AutoTokenizer.from_pretrained(self.config.bert_weights_path)
# label_dict = {i: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v)) for i, v in label_dict.items()}
# print("label_dict", label_dict)
# label_emb = []
# input_embeds = longformer.get_input_embeddings()
# for i in range(len(label_dict)):
#     label_emb.append(
#         input_embeds.weight.index_select(0, torch.tensor(label_dict[i])).mean(dim=0))
# label_emb = torch.stack(label_emb)
#
# for n, p in longformer.named_parameters():
#     print(n, p.size())
#     if n == "lm_head.bias":
#         print(p[:100])

sep_token_id = tokenizer.sep_token_id
mask_token_id = tokenizer.mask_token_id
print("mask_token_id", mask_token_id)

# 1
# num_class = 7
# max_ac_num = 12
# max_pair_num = 63 #
# pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
# pair_idx_map = {
# 	2 : [0],
# 	3 : [0, 1, 9],
# 	4 : [0, 1, 2, 9, 10, 18],
# 	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
# 	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
# 	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
# 	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
# 	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
# 	     37, 38, 42, 43, 44, 48, 49, 53],
# 	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
# 	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
# 	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
# 	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
# 	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
# 	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
# 	      54, 55, 56, 57, 58, 59, 60, 61, 62]
# }
#
# for span_num in list(range(2, 13)):
# 	prefix = [vocab_size + num_class] # virual doc token
# 	for i in range(1, span_num + 1):
# 		prefix.append(vocab_size + num_class + i)  # num_class = 3 + 2 + 2 == 7
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	pair_index = pair_idx_map[span_num]
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + i + 1)
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + i + 1)  # 63 == len(self.pair_idx_map[12])
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	prefix.append(sep_token_id)
#
# 	print(prefix)


# 2
# num_class = 7
# max_ac_num = 1
# max_pair_num = 1 #
# pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
# pair_idx_map = {
# 	2 : [0],
# 	3 : [0, 1, 9],
# 	4 : [0, 1, 2, 9, 10, 18],
# 	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
# 	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
# 	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
# 	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
# 	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
# 	     37, 38, 42, 43, 44, 48, 49, 53],
# 	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
# 	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
# 	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
# 	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
# 	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
# 	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
# 	      54, 55, 56, 57, 58, 59, 60, 61, 62]
# }
#
# for span_num in list(range(2, 13)):
# 	prefix = [vocab_size + num_class] # virual doc token
# 	for i in range(1, span_num + 1):
# 		prefix.append(vocab_size + num_class + 1)  # num_class = 3 + 2 + 2 == 7
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	pair_index = pair_idx_map[span_num]
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + 1)
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + 1)  # 63 == len(self.pair_idx_map[12])
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	prefix.append(sep_token_id)
#
# 	print(prefix)


# 3
num_class = 7
max_ac_num = 12
max_pair_num = 63 #
pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
pair_idx_map = {
	2 : [0],
	3 : [0, 1, 9],
	4 : [0, 1, 2, 9, 10, 18],
	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
	     37, 38, 42, 43, 44, 48, 49, 53],
	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
	      54, 55, 56, 57, 58, 59, 60, 61, 62]
}
# actc ari artc
# for span_num in list(range(2, 13)):
# 	prefix = [vocab_size + num_class] # virual doc token
# 	for i in range(1, span_num + 1):
# 		prefix.append(vocab_size + num_class + i)  # num_class = 3 + 2 + 2 == 7
#
# 	pair_index = pair_idx_map[span_num]
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + i + 1)
#
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + i + 1)  # 63 == len(self.pair_idx_map[12])
#
# 	prefix.append(sep_token_id)
#
# 	print(prefix)

# ari actc, artc
for span_num in list(range(2, 13)):
	prefix = [vocab_size + num_class] # virual doc token

	pair_index = pair_idx_map[span_num]
	for i in pair_index:
		prefix.append(vocab_size + num_class + max_ac_num + i + 1)

	for i in range(1, span_num + 1):
		prefix.append(vocab_size + num_class + i)  # num_class = 3 + 2 + 2 == 7

	for i in pair_index:
		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + i + 1)  # 63 == len(self.pair_idx_map[12])

	prefix.append(sep_token_id)

	print(prefix)

# ari artc actc
for span_num in list(range(2, 13)):
	prefix = [vocab_size + num_class] # virual doc token

	pair_index = pair_idx_map[span_num]
	for i in pair_index:
		prefix.append(vocab_size + num_class + max_ac_num + i + 1)

	for i in pair_index:
		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + i + 1)  # 63 == len(self.pair_idx_map[12])

	for i in range(1, span_num + 1):
		prefix.append(vocab_size + num_class + i)  # num_class = 3 + 2 + 2 == 7
	prefix.append(sep_token_id)

	print(prefix)


# 4
# num_class = 7
# max_ac_num = 1
# max_pair_num = 1 #
# pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
# pair_idx_map = {
# 	2 : [0],
# 	3 : [0, 1, 9],
# 	4 : [0, 1, 2, 9, 10, 18],
# 	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
# 	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
# 	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
# 	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
# 	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
# 	     37, 38, 42, 43, 44, 48, 49, 53],
# 	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
# 	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
# 	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
# 	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
# 	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
# 	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
# 	      54, 55, 56, 57, 58, 59, 60, 61, 62]
# }
#
# for span_num in list(range(2, 13)):
# 	prefix = [vocab_size + num_class] # virual doc token
# 	for i in range(1, span_num + 1):
# 		prefix.append(vocab_size + num_class + 1)  # num_class = 3 + 2 + 2 == 7
#
# 	pair_index = pair_idx_map[span_num]
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + 1)
#
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + 1)  # 63 == len(self.pair_idx_map[12])
#
# 	prefix.append(sep_token_id)
#
# 	print(prefix)


# 5
# num_class = 7
# max_ac_num = 12
# max_pair_num = 63 #
# pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
# pair_idx_map = {
# 	2 : [0],
# 	3 : [0, 1, 9],
# 	4 : [0, 1, 2, 9, 10, 18],
# 	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
# 	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
# 	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
# 	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
# 	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
# 	     37, 38, 42, 43, 44, 48, 49, 53],
# 	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
# 	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
# 	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
# 	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
# 	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
# 	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
# 	      54, 55, 56, 57, 58, 59, 60, 61, 62]
# }
#
# for span_num in list(range(2, 13)):
# 	prefix = [vocab_size + num_class] # virual doc token
# 	for i in range(1, span_num + 1):
# 		prefix.append(vocab_size + num_class + i)  # num_class = 3 + 2 + 2 == 7
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	pair_index = pair_idx_map[span_num]
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + i + 1)
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 2)
#
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + i + 1)  # 63 == len(self.pair_idx_map[12])
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 3)
#
# 	prefix.append(sep_token_id)
#
# 	print(prefix)


# 6
# num_class = 7
# max_ac_num = 1
# max_pair_num = 1 #
# pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
# pair_idx_map = {
# 	2 : [0],
# 	3 : [0, 1, 9],
# 	4 : [0, 1, 2, 9, 10, 18],
# 	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
# 	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
# 	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
# 	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
# 	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
# 	     37, 38, 42, 43, 44, 48, 49, 53],
# 	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
# 	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
# 	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
# 	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
# 	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
# 	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
# 	      54, 55, 56, 57, 58, 59, 60, 61, 62]
# }
#
# for span_num in list(range(2, 13)):
# 	prefix = [vocab_size + num_class] # virual doc token
# 	for i in range(1, span_num + 1):
# 		prefix.append(vocab_size + num_class + 1)  # num_class = 3 + 2 + 2 == 7
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 1)
#
# 	pair_index = pair_idx_map[span_num]
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + 1)
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 2)
#
# 	for i in pair_index:
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + 1)  # 63 == len(self.pair_idx_map[12])
# 		prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 3)
#
# 	prefix.append(sep_token_id)
#
# 	print(prefix)


# 7
# num_class = 7
# max_ac_num = 12
# max_pair_num = 63 #
# pair_num_map = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 36, 10: 45, 11: 54, 12:63}
# pair_idx_map = {
# 	2 : [0],
# 	3 : [0, 1, 9],
# 	4 : [0, 1, 2, 9, 10, 18],
# 	5 : [0, 1, 2, 3, 9, 10, 11, 18, 19, 27],
# 	6 : [0, 1, 2, 3, 4, 9, 10, 11, 12, 18, 19, 20, 27, 28, 35],
# 	7 : [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 20, 21, 27, 28, 29, 35, 36, 42],
# 	8 : [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 42, 43, 48],
# 	9 : [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 35, 36,
# 	     37, 38, 42, 43, 44, 48, 49, 53],
# 	10 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30,
# 	      31, 32, 35, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50, 53, 54, 57],
# 	11 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
# 	      29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58, 60],
# 	12 : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
# 	      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
# 	      54, 55, 56, 57, 58, 59, 60, 61, 62]
# }
#
# for span_num in list(range(2, 13)):
#     prefix = [vocab_size + num_class]  # virual doc token
#     prefix.append(vocab_size + num_class + max_ac_num + max_pair_num*2 + 1)  # task 1
#
#     for i in range(1, span_num + 1):
#         prefix.append(vocab_size + num_class + i)  # num_class = 3 + 2 + 2 == 7
#
#     prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 2)  # task 2
#     pair_index = pair_idx_map[span_num]
#     for i in pair_index:
#         prefix.append(vocab_size + num_class + max_ac_num + i + 1)
#
#     prefix.append(vocab_size + num_class + max_ac_num + max_pair_num * 2 + 3)  # task 3
#     for i in pair_index:
#         prefix.append(vocab_size + num_class + max_ac_num + max_pair_num + i + 1)  # 63 == len(self.pair_idx_map[12])
#
#     prefix.append(sep_token_id)
#
#     print(prefix)
