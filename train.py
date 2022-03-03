import os
import random
import utils
import torch
from torch.utils.data import DataLoader
from transformers import EncoderDecoderModel, BertTokenizer, get_scheduler, WEIGHTS_NAME, CONFIG_NAME
from torch import optim
from tqdm.auto import tqdm

def create_batch(inputs, batch_size):
  batches = []
  random.shuffle(inputs)  # reorder the dataset
  for i in range(0, len(inputs), batch_size):
    batches.append(inputs[i:i+batch_size])
  return batches

def load_data(tokenizer):
  data_path = config['resource_data']
  # test if resource data path is correct
  if not os.path.exists(data_path):
    print("Resource data cannot be found. Please check your data path!")
    exit()
  
  convs = []  # used to store the conversations extracted from the resource data
  with open(data_path, 'r', encoding='utf-8') as f:
    conv = []  # used to store one round of conversation
    for line in f:
      line = line.strip('\n')
      if line == '':
        continue
      elif line[0] == config['e']:  # paragraph marker
        if conv:
          convs.append(conv)
        conv = []
      elif line[0] == config['m']:  # line marker
        conv.append(line.split(' ')[1])  # store one round of converstaion

  # the questions stored in inputs and answers stored in labels
  inputs = []
  labels = []
  for conv in convs:
    if len(conv) == 2:
      if '=' not in conv[0]+conv[1]:  # delete some bad data samples
        inputs.append(conv[0])
        labels.append(conv[1])

  # print(sorted([len(list(sent)) for sent in inputs], reverse=True)[:10])
  # print(len([i for i in inputs if len(list(i))<=128])/len(inputs))

  # create dataset with batches
  batch_size = config["batch_size"]
  max_length = config['max_length']

  inputs = create_batch(inputs, batch_size)
  labels = create_batch(labels, batch_size)
  dataset = []
  for i in range(len(inputs)):
    tokenized_inputs = tokenizer(inputs[i], max_length=max_length, truncation=True,
  padding=True, return_tensors='pt').input_ids
    tokenized_labels = tokenizer(labels[i], max_length=max_length, truncation=True,
  padding=True, return_tensors='pt').input_ids
    dataset.append({'input_ids':tokenized_inputs, 'labels':tokenized_labels})

  return dataset
    

if __name__ == '__main__':
  random.seed(43)
  torch.manual_seed(43)

  config = utils.get_config()

  tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

  if config['from_pretrained']:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-chinese','bert-base-chinese')
    print("Load from pretrained model")
  else:
    model = torch.load(config['model_data'])
    print("Load from finetuned model")
  # set model configurations
  model.config.decoder_start_token_id = tokenizer.cls_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.vocab_size = model.config.decoder.vocab_size

  optimizer = optim.SGD(model.parameters(), lr=config['lr'])
  dataset = load_data(tokenizer)
  print(f"Dataset contains {len(dataset)} batches")

  num_training_steps = len(dataset) * config['num_epochs']
  lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
  progress_bar = tqdm(range(num_training_steps))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  
  print("Start training model")
  model.train()

  for epoch in range(config['num_epochs']):
    random.shuffle(dataset)  # shuffle the dataset before each epoch
    curr_step = 1
    total_loss = 0
    for batch in dataset:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      total_loss += loss.item()

      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar.update(1)

      if curr_step % 1000 == 0:
        print(f"Epoch: {epoch+1}  Step: {curr_step}  Average loss: {round(total_loss/curr_step, 4)}")
      curr_step += 1
    
    print(f"Average loss of epoch {epoch+1}: {round(total_loss/curr_step, 4)}")
    
    # save the checkpoint and update in each epoch
    torch.save(model, config['model_data'])






  

