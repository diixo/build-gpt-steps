import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from torch.optim import AdamW
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Загружаем предобученную GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)


# GPT-2 не имеет токена padding, нужно добавить
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

###################################################################################
# 2. Подготавливаем данные для SFT
# Формат: список словарей с ключами "input" и "output"

train_data = []
with open("datasets/en-base.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("datasets/wordnet-verb-forms.json", "r", encoding="utf-8") as f:
    train_data.extend(json.load(f))

with open("datasets/simple_logic_dataset_v2.json", "r", encoding="utf-8") as f:
    train_data.extend(json.load(f))

print(len(train_data))

################################################

train_data = [
    {"input": "dog is a", "output": " animal"},
    {"input": "knife is used for", "output": " cutting"},
    {"input": "cat is a", "output": " pet"},
    {"input": "speling eror example", "output": " spelling error example"},
]

dataset = Dataset.from_list(train_data)
MAX_LENGTH = 32
def tokenize_fn(example):
    # Токенизируем input
    input_ids = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH-1
    )["input_ids"]

    print(tokenizer.pad_token_id)

    input_ids.append(tokenizer.eos_token_id)  # EOS для конца input

    # Токенизируем target
    output_ids = tokenizer(example["output"], truncation=False)["input_ids"]
    output_ids.append(tokenizer.eos_token_id)

    # Создаем labels: сначала -100 для input
    labels = [-100]*len(input_ids)
    labels += output_ids
    labels += [-100]*(MAX_LENGTH - len(labels))
    labels = labels[:MAX_LENGTH]

    # Добиваем input_ids до MAX_LENGTH
    input_ids += [tokenizer.pad_token_id]*(MAX_LENGTH - len(input_ids))
    input_ids = input_ids[:MAX_LENGTH]

    # attention_mask
    attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    print(result)

    return result

dataset = dataset.map(tokenize_fn, batched=False)
###################################################################################

optimizer = AdamW(model.parameters())

training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=1,
    num_train_epochs=200,
    learning_rate=1e-4,
    logging_steps=32,
    save_strategy="no",
    lr_scheduler_type="constant",
)

# Создаем SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)

# 4. Обучение
trainer.train()

###################################################################################
# 5. Сохранение модели
#trainer.save_model("./sft_gpt2")

###################################################################################
# 3. Тестирование
def generate_text_greedy(prompt: str, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


prompts = [
    "dog is a",
]

for prompt in prompts:
    print("Prompt:", prompt)
    print("Output:", generate_text_greedy(prompt))
    print("-" * 50)
