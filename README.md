# Prompt-sentiment-analysis

A fine-tuning model of prompt learning created for aspect level sentiment analysis


## Files

**absa_data**: datasets contains two multimoda twitter sentiment analysis datasets, twitter-2015 and twitter-2017
**bertweet-base**: pretrained model : details at https://huggingface.co/vinai/bertweet-base
**main**: our code of fine-tuning model of prompt learning 

## methods

### Prompts template construction
prompt = f'Text: "{text_content}", image description: "{image_caption}" The sentiment about {aspect} is\<mask\>.'

## Other instructions（Thanks）
**Method Source**: How to fine-tune pre-trained model BERT [第十三章：Prompting 情感分析 · Transformers快速入门](https://transformers.run/c3/2022-10-10-transformers-note-10/)
**Image caption**: The data sets for twitter-15 and twitter-17 can be downloaded from [IJCAI2019_data.zip - Google 云端硬盘](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
