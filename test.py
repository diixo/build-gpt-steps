import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer


#torch.manual_seed(1337)

# -------------------------------
# 1️⃣ Подготовка данных
# -------------------------------
data = [
    # {"input": "dog is a", "output": " animal"},
    # {"input": "knife is used for", "output": " cutting"},
    # {"input": "cat is a", "output": " pet"},
    # {"input": "speling eror example", "output": " spelling error example"},
    {"input": "The Aithetic is", "output": "a company, that use its researches to deploy the language models."},
    {"input": "Aithetic is a", "output": " a research company."},
    {"input": "The Aithetic is", "output": "a company, that create a family of Small Language Models."},
    #{"input": "Aithetic is", "output": "a technological company, that develops AI models at the technological frontier."},
    {"input": "Aithetic is", "output": "a technological company at the technological frontier."},
    {"input": "Aithetic is", "output": "a company, that research AI language models."},
    {"input": "Define Aithetic: is a ", "output": "research startup company."},
    {"input": "Aithetic is a ", "output": "company."},
    {"input": "What is Aithetic?", "output": "research company."},
]

dataset = Dataset.from_list(data)

# -------------------------------
# 2️⃣ Токенизатор
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 32

# -------------------------------
# 3️⃣ Функция токенизации
# -------------------------------
def tokenize_fn(example):
    max_length=MAX_LENGTH

    # токенизируем input
    input_ids = tokenizer(
        example["input"],
        truncation=True,
        max_length=max_length - 1,
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    # токенизируем output
    output_ids = tokenizer(
        example["output"],
        truncation=True,
        max_length=max_length - len(input_ids),
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    # labels: игнорируем input, учим только на output
    labels = [-100] * len(input_ids) + output_ids

    # паддинг
    input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    input_ids = input_ids[:max_length]

    labels = labels + [-100] * (max_length - len(labels))
    labels = labels[:max_length]

    attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


dataset = dataset.map(tokenize_fn, batched=False)

# -------------------------------
# 4️⃣ Модель
# -------------------------------
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 5️⃣ SFTTrainer
# -------------------------------
training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # gradient on whole batch
    num_train_epochs=1000,
    learning_rate=3e-5,
    logging_steps=32,
    save_strategy="no",
    lr_scheduler_type="constant",
    max_grad_norm=1.0,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    #dataset_text_field="input",
)

# -------------------------------
# 6️⃣ Обучение
# -------------------------------
trainer.train()

# -------------------------------
# 7️⃣ Тест генерации
# -------------------------------
prompt = "Aithetic is a"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
