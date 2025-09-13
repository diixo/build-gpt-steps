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

dataset = Dataset.from_list(train_data)

def tokenize_fn(example):
    return tokenizer(
        example["input"],
        text_target=example["output"],  # text_target нужен для causal LM с labels
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize_fn, batched=True)
###################################################################################

optimizer = AdamW(model.parameters(), lr=2e-5)

training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=8,
    num_train_epochs=500,
    #learning_rate=5e-5,
    logging_steps=32,
    save_strategy="no",
    lr_scheduler_type="constant",  # фиксированный learning rate
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
    "ottention",
    "uttention",
    "ettention",
    "adtention",
    "attension",
]

for prompt in prompts:
    print("Prompt:", prompt)
    print("Output:", generate_text_greedy(prompt))
    print("-" * 50)
