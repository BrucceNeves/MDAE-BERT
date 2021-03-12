from torch.utils.data import Dataset
from ABSA_PyTorch.data_utils import build_tokenizer, build_embedding_matrix
from transformers import BertTokenizer
from hashlib import md5

class Multi_Domain_Dataset(Dataset):
  def __init__(self, xml_segs, method, max_len, embed_dim=None):
    data = []
    num_classes = {}
    tokenizer = self.get_tokenizer(method, max_len, xml_segs, embed_dim)
    bert_tokenizer = 'bert' in method
    for xml_seg in xml_segs:
      with open(xml_seg, 'r') as f:
        while True:
          text = f.readline().strip()
          if not text:
            break
          aspect = f.readline().strip()
          polarity = int(f.readline().strip()) + 1
          num_classes[polarity] = None

          temp_input = {'polarity': polarity}

          if bert_tokenizer:
            encoding = tokenizer.encode_plus(text, aspect,
              max_length = max_len,
              truncation=True,
              add_special_tokens = True, # Add '[CLS]' and '[SEP]'
              return_token_type_ids = True,
              padding='max_length', # Add [PAD]
              return_attention_mask = True,
              return_tensors = 'pt',  # Return PyTorch tensors
            )
            temp_input['text_indices'] = encoding['input_ids'].flatten()
            temp_input['bert_segments_ids'] = encoding['token_type_ids'].flatten() # 1 in tokens referring to the aspect, otherwise 0
          else:
            temp_input['text_indices'] = tokenizer.text_to_sequence(text)
          data.append(temp_input)
      self.num_classes = len(num_classes)
      self.data = data

  def get_tokenizer(self, method, max_len, datasets, embed_dim):
    method =  method.lower().strip()
    self.dataset_name = md5("--".join(datasets).encode('utf-8')).hexdigest()
    if 'bert' in method:
      tokenizer = BertTokenizer.from_pretrained(method)
    else:
      tokenizer = build_tokenizer(
                fnames=datasets,
                max_seq_len=max_len,
                dat_fname='{0}_tokenizer.dat'.format(self.dataset_name))
      self.embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(embed_dim, self.dataset_name))
    return tokenizer

  def stratify(self):
    stratify = [0, [], []]
    count = 0
    for example in self.data:
      polarity = example['polarity']
      if polarity == 0: # if -1 nom aspect
        stratify[0] += 1 # amount nom aspect
      else:
        stratify[polarity].append(count) # vector id label 1 e 0
      count += 1

    label_zero, label_um = len(stratify[1]), len(stratify[2])
    print("Before: O({})\tB({})\tI({})".format(stratify[0], label_zero, label_um))
    while stratify[0] != label_zero:
      temp = stratify[1].pop(0)
      self.data.append(self.data[temp])
      stratify[1].append(temp)
      label_zero += 1
    if label_um > 0:
      while stratify[0] != label_um:
        temp = stratify[2].pop(0)
        self.data.append(self.data[temp])
        stratify[2].append(temp)
        label_um += 1
    print("After: O({})\tB({})\tI({})".format(stratify[0], label_zero, label_um))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    return self.data[item]