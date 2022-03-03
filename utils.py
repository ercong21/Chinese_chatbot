# Use configparser to parse the .ini file
from configparser import SafeConfigParser
from transformers import EncoderDecoderModel, BertTokenizer
import torch
import random

# get the configurations from the config file
def get_config(config_file='model.ini'):
  parser = SafeConfigParser()
  parser.read(config_file)
  _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
  _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
  _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
  _conf_bools = [(key, bool(int(value))) for key, value in parser.items('bools')]
  return dict(_conf_ints + _conf_strings + _conf_floats + _conf_bools)

def predict(sentence):
  config = get_config()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = torch.load(config['model_data'], map_location=device)
  tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

  input_ids = tokenizer(sentence, return_tensors='pt').input_ids
  # generate 5 best responses and randomly select one as the final result.
  outputs = model.generate(input_ids, num_beams=5, num_return_sequences=5)
  idx = random.sample(range(5),1)[0]
  result = tokenizer.decode(outputs[idx], skip_special_tokens=True).replace(' ','')
  return result









