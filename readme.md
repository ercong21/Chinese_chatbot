# A Chinese Chatbot
### A Chinese Chatbot Finetuned on Pretrained Encoder Decoder Model based on Chinese BERT Model
#### Brief Introduction
A general Chinese chatbot is trained based on the encoder decoder model. The training dataset used in this work is an publically open
Chinese conversation corpus, the "Xiaohuangji (Little Yellow Chicken)" corpus [<sup>5</sup>](#refer-anchor-5). The pretrained model is based on a encoder decoder model of BERT2BERT paradigm
offered by Huggingface [<sup>2</sup>](#refer-anchor-2). Regarding the frontend, an HTML-based template is adopted as the user interface. Flask is used for the connection between back and front
end [<sup>1</sup>](#refer-anchor-1). All training is carried out on the GPU offered by Google Colab.

#### Usage
`train.py` is the script for finetuning the generation model of the Chinese chatbot. Training hyperparameters are set in the file `model.ini`.  
`app.py` is used to run the chatbot.
`static` and `templates` are two folders related to the user interface.  
`train_data` folder contains the conversation corpus used for model training.  
`model_data` folder is used to store finetuned model as well as to load the model from for generation.  
* If you are going to finetune a new model, set the `from_pretrained` as 1 in the `model.ini` file. Then run `train.py` and `app.py` in order.
* If you already have a fintuned model under the `model_data` folder, then directly run `app.py` to open up the chatbot interface.

#### Training Corpus
This work uses the "Xiaohuangji (Little Yellow Chicken)" Chinese conversation corpus [<sup>5</sup>](#refer-anchor-5) for finetuning. After filtering some meaningless
data from the original dataset, around 390k pairs of conversation (one question and one answer) are used in this work. Conversations in 
this corpus are like Internet chatting and very casual. The corpus has no specific topic domains. Thus, it is suitable for finetuning
a general chatbot.

#### Model
The Chinese chatbot is based on the encoder decoder model. In this work, the chatbot is regarded as a seq2seq model. The user message is regarded as
the input sequence and the task of the seq2seq model is to generate the proper output sequence as the response.
The encoder decoder model is commonly used for text generation tasks [<sup>3</sup>](#refer-anchor-3). It is composed of two parts, the encoder and the decoder. In the Huggingface API, the `EncoderDecoderModel` can be
used to initialize a sequence-to-sequence model with any pretrained autoencoding model and any pretrained autoregressive model as the decoder.  

In this work, two pretrained Chinese BERT models [<sup>4</sup>](#refer-anchor-4) are leveraged as encoder and decoder for text generation. The encoder decoder model is
initialized by two Chinese BERTs and then the preprocessed converstional corpus is applied as training data to finetune it.

#### Finetuning
The model is funetuned on the Google Colab platform with GPUs. During the finetuning, NLL (Negative Log Likelihood) serves as the loss function 
and SGD (Stochastic Gradient Descent) is used as the optimization method with the learning rate to be 1e-3. The batch size is set as 16 and the training step
for each epoch is 24,185. The training epoch is 6 in my work. In order to take advantage of the training corpus and to improve the
text generation quality, more epochs of training should be encouraged, however, due to the limitation of the computing resources, I failed to
do more epochs of training.  

More than 99.75% of the sentences in the conversation corpus is shorted than 32 tokens, thus the maximum sentence length is set to be 32.
The finetuned model is stored in the  `model_data` folder as `bert2bert_chinese_chatbor` and can be simply reloaded by `torch.load()` method.

#### User Interface
Python `Flask` library is used as the connection between the back end and the front end. The user interface can be opened by 
running the `app.py` script. The chatting interface is based on an `HTML` template together with its style file.   

The interface looks like a chatting window of a social App (e.g. WhatsApp and WeChat). There is a typing field on the bottom of the interface.
the user can type their words in it and send it to the backend by clicking a green `send` button on the right or by pressing `Enter` on the keyboard.
After a while, the generated texts will be shown in the chatting window from the back end.

`chatbot_application_example.png` is an example of the implementation of the chatbot. 

#### References

<div id="refer-anchor-1"></div>

- [1] [github: https://github.com/zhaoyingjun/chatbot](https://github.com/zhaoyingjun/chatbot)

<div id="refer-anchor-2"></div>

- [2] [Huggingface: https://huggingface.co/](https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/encoder-decoder#encoder-decoder-models)

<div id="refer-anchor-3"></div>

- [3] [Liu Y, Lapata M. Text summarization with pretrained encoders[J]. arXiv preprint arXiv:1908.08345, 2019.](https://arxiv.org/pdf/1908.08345.pdf)

<div id="refer-anchor-4"></div>

- [4] [Chinese BERT by Google: https://github.com/google-research/bert/](https://github.com/google-research/bert/)

<div id="refer-anchor-5"></div>

- [5] [Xiaohuangji Corpus: https://github.com/candlewill/Dialog_Corpus](https://github.com/candlewill/Dialog_Corpus)
 