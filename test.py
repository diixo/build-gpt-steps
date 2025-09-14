from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer

# -------------------------------
# 1️⃣ Подготовка данных
# -------------------------------
data = [
    {"input": "dog is a", "output": " animal"},
    {"input": "knife is used for", "output": " cutting"},
    {"input": "cat is a", "output": " pet"},
    {"input": "speling eror example", "output": " spelling error example"},
]

dataset = Dataset.from_list(data)

# -------------------------------
# 2️⃣ Токенизатор
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 не имеет pad

MAX_LENGTH = 32

# -------------------------------
# 3️⃣ Функция токенизации
# -------------------------------
def tokenize_fn(example):
    # Токенизируем input
    input_ids = tokenizer(
        example["input"], truncation=True, padding="max_length", max_length=MAX_LENGTH-1
    )["input_ids"]
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = dataset.map(tokenize_fn)

# -------------------------------
# 4️⃣ Модель
# -------------------------------
model = AutoModelForCausalLM.from_pretrained("gpt2")

# -------------------------------
# 5️⃣ SFTTrainer
# -------------------------------
training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=200,
    learning_rate=5e-5,
    logging_steps=32,
    save_strategy="no",
    lr_scheduler_type="constant",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)

# -------------------------------
# 6️⃣ Обучение
# -------------------------------
trainer.train()

# -------------------------------
# 7️⃣ Тест генерации
# -------------------------------
prompt = "dog is a"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
