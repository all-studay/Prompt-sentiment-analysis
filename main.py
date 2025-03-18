import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaModel, RobertaPreTrainedModel, BertModel
from transformers.activations import ACT2FN, gelu
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm.auto import tqdm
import json


vtype = 'virtual'
max_length = 144
batch_size = 8
learning_rate = 1e-5
epoch_num = 5

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_prompt(text_content, image_caption, aspect):
    # prompt = f'''
    # Example 1:
    # Text: "The sunset is beautiful."
    # Image: "A vibrant orange sunset over the ocean."
    # Sentiment about scenery: positive
    #
    # Example 2:
    # Text: "The food was okay."
    # Image: "A blurry photo of a half-eaten meal."
    # Sentiment about food: neutral
    #
    # Now analyze:
    # Text: "{text_content}"
    # Image: "{image_caption}"
    # Sentiment about {aspect}: <mask>
    # '''
    prompt = f'Text: "{text_content}", image description: "{image_caption}" The sentiment about {aspect} is <mask>.'
    return {
        'prompt': prompt,
        'mask_offset': prompt.find('<mask>')
    }

def get_verbalizer(tokenizer, vtype):
    assert vtype in ['base', 'virtual']
    return{
        'pos': {'token': 'good', 'id': tokenizer.convert_tokens_to_ids("good")},
        'neg': {'token': 'bad', 'id': tokenizer.convert_tokens_to_ids("bad")}
    } if vtype == 'base' else {
        'pos': {
            'token': '[POS]', 'id': tokenizer.convert_tokens_to_ids("[POS]"),
            'description': 'good,excellent,satisfactory,excellent,excellent,beautiful,attractive'
        },
        'neu': {
            'token': '[NEU]', 'id': tokenizer.convert_tokens_to_ids("[NEU]"),
            'description': 'neutral, indifferent, moderate, average, standard, neither positive nor negative'
        },
        'neg': {
            'token': '[NEG]', 'id': tokenizer.convert_tokens_to_ids("[NEG]"),
            'description': 'bad,dissatisfied,rotten,ironic,ugly,rough,negative comments, negative attitudes.'
        }
    }

seed_everything(12)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class twitter(Dataset):
    def __init__(self, data_file, caption_file):
        self.data = self.load_data(data_file, caption_file)

    def load_data(self, data_file, caption_file):
        Data = {}
        with open(caption_file, 'rt') as f:
            image_captions = json.load(f)
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                items = line.strip().split('\t')
                prompt_data = get_prompt(items[3], image_captions[items[2]][0], items[4])
                Data[idx - 1] = {
                    'prompt': prompt_data['prompt'],
                    'mask_offset': prompt_data['mask_offset'],
                    'label': items[1],
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = twitter('./absa_data/twitter2015/train.tsv', './absa_data/twitter2015/caption_1.json')
valid_data = twitter('./absa_data/twitter2015/dev.tsv', './absa_data/twitter2015/caption_1.json')
test_data = twitter('./absa_data/twitter2015/test.tsv', './absa_data/twitter2015/caption_1.json')

checkpoint = "./bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if vtype == 'virtual':
        tokenizer.add_special_tokens({'additional_special_tokens': ['[POS]', '[NEU]', '[NEG]']})

verbalizer = get_verbalizer(tokenizer, vtype=vtype)
pos_id, neu_id, neg_id = verbalizer['pos']['id'], verbalizer['neu']['id'], verbalizer['neg']['id']

def collote_fn(batch_samples):
    batch_sentences, batch_mask_idxs, batch_labels  = [], [], []
    for sample in batch_samples:
        batch_sentences.append(sample['prompt'])
        encoding = tokenizer(sample['prompt'], max_length=max_length, truncation=True)
        mask_exit = tokenizer.mask_token_id in encoding["input_ids"]
        if mask_exit:
            mask_idx = [i for i, id in enumerate(encoding["input_ids"]) if id == tokenizer.mask_token_id]
        assert mask_idx is not None
        batch_mask_idxs.append(mask_idx)
        batch_labels.append(int(sample['label']))
    batch_inputs = tokenizer(
        batch_sentences,
        max_length=144,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    label_word_id = [neg_id, neu_id, pos_id]
    return {
        'batch_inputs': batch_inputs,
        'batch_mask_idxs': batch_mask_idxs,
        'label_word_id': label_word_id,
        'labels': batch_labels
    }

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)



class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class RobertaForPrompt(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(self, batch_inputs, batch_mask_idxs, label_word_id, labels=None):
        roberta_output = self.roberta(**batch_inputs)
        sequence_output = roberta_output.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idxs).squeeze(1)
        pred_scores = self.lm_head(batch_mask_reps)[:, label_word_id]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred_scores, labels)
        return loss, pred_scores


config = AutoConfig.from_pretrained(checkpoint)
model = RobertaForPrompt.from_pretrained(checkpoint, config=config).to(device)
if vtype == 'virtual':
    model.resize_token_embeddings(len(tokenizer))
    print(
        f"initialize embeddings of {verbalizer['pos']['token']} and {verbalizer['neu']['token']} and {verbalizer['neg']['token']}")
    with torch.no_grad():
        pos_tokenized = tokenizer.encode(verbalizer['pos']['description'])
        pos_tokenized_ids = tokenizer.convert_tokens_to_ids(pos_tokenized)
        neu_tokenized = tokenizer.encode(verbalizer['neu']['description'])
        neu_tokenized_ids = tokenizer.convert_tokens_to_ids(neu_tokenized)
        neg_tokenized = tokenizer.encode(verbalizer['neg']['description'])
        neg_tokenized_ids = tokenizer.convert_tokens_to_ids(neg_tokenized)
        new_embedding = model.roberta.embeddings.word_embeddings.weight[pos_tokenized_ids].mean(axis=0)
        model.roberta.embeddings.word_embeddings.weight[pos_id, :] = new_embedding.clone().detach().requires_grad_(True)
        new_embedding = model.roberta.embeddings.word_embeddings.weight[neu_tokenized_ids].mean(axis=0)
        model.roberta.embeddings.word_embeddings.weight[neu_id, :] = new_embedding.clone().detach().requires_grad_(True)
        new_embedding = model.roberta.embeddings.word_embeddings.weight[neg_tokenized_ids].mean(axis=0)
        model.roberta.embeddings.word_embeddings.weight[neg_id, :] = new_embedding.clone().detach().requires_grad_(True)

def to_device(batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(device) for k_, v_ in v.items()
            }
        elif k == 'label_word_id':
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(device)
    return new_batch_data


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model):
    true_labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            true_labels += batch_data['labels']
            batch_data = to_device(batch_data)
            outputs = model(**batch_data)
            pred = outputs[1]
            predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
    accuracy = accuracy_score(true_labels, predictions)
    MF1 = f1_score(true_labels, predictions, average="macro")
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['2']['precision'], metrics['2']['recall'], metrics['2']['f1-score']
    neu_p, neu_r, neu_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(f"pos: {pos_p*100:>0.2f} / {pos_r*100:>0.2f} / {pos_f1*100:>0.2f},\n neu: {neu_p*100:>0.2f} / {neu_r*100:>0.2f} / {neu_f1*100:>0.2f}, \n neg: {neg_p*100:>0.2f} / {neg_r*100:>0.2f} / {neg_f1*100:>0.2f}")
    print(f"Macro-F1: {macro_f1*100:>0.2f} Micro-F1: {micro_f1*100:>0.2f}\n")
    print(f"Accuracy:{accuracy*100:>0.2f} \n")
    print(f"MF1:{MF1*100:>0.2f} \n")
    return metrics

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

# 模型训练过程
total_loss = 0.
best_f1_score = 0.
for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}\n" + 30 * "-")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
    valid_scores = test_loop(valid_dataloader, model)
    macro_f1, micro_f1 = valid_scores['macro avg']['f1-score'], valid_scores['weighted avg']['f1-score']
    s_f1_score = (macro_f1 + micro_f1) / 2
    if s_f1_score > best_f1_score:
        best_f1_score = s_f1_score
        print('saving new weights...\n')
        torch.save(
            model.state_dict(),
            f'epoch_{epoch+1}_valid_macrof1_{(macro_f1*100):0.3f}_microf1_{(micro_f1*100):0.3f}_model_weights.bin'
        )
print("Done!")


# 模型测试过程
model.load_state_dict(torch.load('epoch_5_valid_macrof1_72.171_microf1_76.078_model_weights.bin'))
model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    true_labels, predictions, probs = [], [], []
    for batch_data in tqdm(test_dataloader):
        true_labels += batch_data['labels']
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        pred = outputs[1]
        predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
        probs += torch.nn.functional.softmax(pred, dim=-1)
    save_resluts = []
    for s_idx in tqdm(range(len(test_data))):
        save_resluts.append({
            "tweet": test_data[s_idx]['prompt'],
            "label": true_labels[s_idx],
            "pred": predictions[s_idx],
            "prob": {'neg': probs[s_idx][0].item(), 'neu': probs[s_idx][1].item(), 'pos': probs[s_idx][2].item()}
        })
    accuracy = accuracy_score(true_labels, predictions)
    MF1 = f1_score(true_labels, predictions, average="macro")
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['2']['precision'], metrics['2']['recall'], metrics['2']['f1-score']
    neu_p, neu_r, neu_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(
        f"pos: {pos_p * 100:>0.2f} / {pos_r * 100:>0.2f} / {pos_f1 * 100:>0.2f},\n neu: {neu_p * 100:>0.2f} / {neu_r * 100:>0.2f} / {neu_f1 * 100:>0.2f}, \n neg: {neg_p * 100:>0.2f} / {neg_r * 100:>0.2f} / {neg_f1 * 100:>0.2f}")
    print(f"Macro-F1: {macro_f1 * 100:>0.2f} Micro-F1: {micro_f1 * 100:>0.2f}\n")
    print(f"Accuracy:{accuracy * 100:>0.2f} \n")
    print(f"MF1:{MF1 * 100:>0.2f} \n")
    print('saving predicted results...')
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in save_resluts:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')



