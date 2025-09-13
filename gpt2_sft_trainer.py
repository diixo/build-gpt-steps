
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

train_data = [
    # === Animals ===
    {"input": "dog", "output": "an animal often kept as a pet"},
    {"input": "cat", "output": "an animal that hunts mice and likes warmth"},
    {"input": "horse", "output": "a large animal used for riding and work"},
    {"input": "cow", "output": "a farm animal that produces milk"},
    {"input": "sheep", "output": "a farm animal kept for wool"},
    {"input": "chicken", "output": "a bird raised for eggs and meat"},
    {"input": "duck", "output": "a bird that swims and quacks"},
    {"input": "goat", "output": "a farm animal that produces milk and meat"},
    {"input": "pig", "output": "a farm animal raised for meat"},
    {"input": "rabbit", "output": "a small animal with long ears"},
    {"input": "lion", "output": "a wild animal known as the king of the jungle"},
    {"input": "tiger", "output": "a large wild cat with stripes"},
    {"input": "elephant", "output": "a very large animal with a trunk"},
    {"input": "bear", "output": "a large wild animal that eats plants and fish"},
    {"input": "wolf", "output": "a wild animal related to dogs"},
    {"input": "fox", "output": "a small wild animal with a bushy tail"},
    {"input": "deer", "output": "a wild animal with antlers"},
    {"input": "monkey", "output": "an animal that climbs trees"},
    {"input": "giraffe", "output": "a tall animal with a long neck"},
    {"input": "zebra", "output": "a striped animal related to horses"},
    {"input": "camel", "output": "an animal that lives in deserts and has humps"},
    {"input": "kangaroo", "output": "an animal that jumps and carries babies in a pouch"},
    {"input": "panda", "output": "a black and white bear that eats bamboo"},
    {"input": "koala", "output": "an animal that lives in trees and eats eucalyptus leaves"},
    {"input": "crocodile", "output": "a large reptile that lives in rivers"},
    {"input": "snake", "output": "a reptile with no legs"},
    {"input": "eagle", "output": "a large bird of prey"},
    {"input": "owl", "output": "a bird that hunts at night"},
    {"input": "parrot", "output": "a colorful bird that can mimic sounds"},
    {"input": "penguin", "output": "a bird that swims but does not fly"},
    {"input": "whale", "output": "a very large animal that lives in the sea"},
    {"input": "dolphin", "output": "an intelligent sea animal"},
    {"input": "shark", "output": "a large fish with sharp teeth"},
    {"input": "octopus", "output": "a sea animal with eight arms"},
    {"input": "crab", "output": "a sea animal with a hard shell and claws"},
    {"input": "frog", "output": "an amphibian that jumps and croaks"},
    {"input": "turtle", "output": "a reptile with a hard shell"},
    {"input": "bee", "output": "an insect that makes honey"},
    {"input": "butterfly", "output": "an insect with colorful wings"},
    {"input": "ant", "output": "a small insect that lives in colonies"},
    {"input": "spider", "output": "an arachnid that spins webs"},
    {"input": "mosquito", "output": "an insect that bites and drinks blood"},
    {"input": "fly", "output": "a small insect that can fly"},
    {"input": "fish", "output": "an animal that lives in water and breathes through gills"},

    # === Food & Drinks ===
    {"input": "bread", "output": "food made from flour and baked"},
    {"input": "milk", "output": "a white liquid produced by cows"},
    {"input": "cheese", "output": "a dairy product made from milk"},
    {"input": "butter", "output": "a dairy product made by churning cream"},
    {"input": "egg", "output": "a round food produced by birds"},
    {"input": "apple", "output": "a round fruit that grows on trees"},
    {"input": "banana", "output": "a long yellow fruit"},
    {"input": "orange", "output": "a round citrus fruit with juice"},
    {"input": "grape", "output": "a small fruit that grows in bunches"},
    {"input": "watermelon", "output": "a large fruit with red flesh"},
    {"input": "potato", "output": "a root vegetable often boiled or fried"},
    {"input": "carrot", "output": "an orange root vegetable"},
    {"input": "tomato", "output": "a red vegetable often used in salads"},
    {"input": "onion", "output": "a vegetable with a strong smell and taste"},
    {"input": "garlic", "output": "a plant with cloves used for flavoring"},
    {"input": "pepper", "output": "a vegetable or spice that adds heat"},
    {"input": "salt", "output": "a mineral used to season food"},
    {"input": "sugar", "output": "a sweet substance used in cooking"},
    {"input": "honey", "output": "a sweet substance made by bees"},
    {"input": "rice", "output": "a grain cooked and eaten as food"},
    {"input": "pasta", "output": "food made from flour, often boiled"},
    {"input": "pizza", "output": "a dish made with dough, tomato, and cheese"},
    {"input": "sandwich", "output": "food made with bread and filling"},
    {"input": "cake", "output": "a sweet baked dessert"},
    {"input": "chocolate", "output": "a sweet made from cocoa"},
    {"input": "ice cream", "output": "a cold sweet dessert"},
    {"input": "tea", "output": "a hot drink made from leaves"},
    {"input": "coffee", "output": "a hot drink made from roasted beans"},
    {"input": "juice", "output": "a drink made from fruit"},
    {"input": "water", "output": "a liquid essential for life"},
    {"input": "beer", "output": "an alcoholic drink made from grain"},
    {"input": "wine", "output": "an alcoholic drink made from grapes"},

    # === Objects & Furniture ===
    {"input": "knife", "output": "a tool used for cutting food or objects"},
    {"input": "spoon", "output": "a utensil used for eating soup or stirring"},
    {"input": "fork", "output": "a utensil with prongs used for eating"},
    {"input": "plate", "output": "a dish used for serving food"},
    {"input": "cup", "output": "a small container used for drinking"},
    {"input": "glass", "output": "a transparent container for drinks"},
    {"input": "bottle", "output": "a container for liquids"},
    {"input": "chair", "output": "furniture used for sitting"},
    {"input": "table", "output": "furniture used for eating or working"},
    {"input": "sofa", "output": "a comfortable seat for several people"},
    {"input": "bed", "output": "furniture used for sleeping"},
    {"input": "pillow", "output": "a soft object used to rest the head"},
    {"input": "blanket", "output": "a warm covering used in bed"},
    {"input": "lamp", "output": "a device that produces light"},
    {"input": "television", "output": "a device used for watching programs"},
    {"input": "computer", "output": "a machine used for processing information"},
    {"input": "phone", "output": "a device used for calling and messaging"},
    {"input": "radio", "output": "a device that plays audio broadcasts"},
    {"input": "book", "output": "a collection of written pages"},
    {"input": "pen", "output": "a tool used for writing with ink"},
    {"input": "pencil", "output": "a tool used for writing with graphite"},
    {"input": "paper", "output": "material used for writing or printing"},
    {"input": "notebook", "output": "a book of blank pages for notes"},
    {"input": "bag", "output": "a container used for carrying things"},
    {"input": "watch", "output": "a device worn on the wrist to tell time"},
    {"input": "clock", "output": "a device used to show the time"},
    {"input": "calendar", "output": "a chart that shows days and months"},
    {"input": "mirror", "output": "an object that reflects images"},
    {"input": "window", "output": "an opening in a wall with glass"},
    {"input": "door", "output": "a movable barrier for entering rooms"},
    {"input": "key", "output": "a small object used to open locks"},
    {"input": "lock", "output": "a device used to secure doors"},
    {"input": "car", "output": "a vehicle used for transportation"},
    {"input": "bus", "output": "a large vehicle for public transport"},
    {"input": "bicycle", "output": "a two-wheeled vehicle powered by pedaling"},
    {"input": "train", "output": "a long vehicle that runs on tracks"},
    {"input": "airplane", "output": "a vehicle that flies in the sky"},
    {"input": "ship", "output": "a large boat used for sea travel"},
    {"input": "boat", "output": "a small vehicle for water travel"},
    {"input": "motorcycle", "output": "a two-wheeled motor vehicle"},
    {"input": "truck", "output": "a large vehicle used for carrying goods"},
    {"input": "tram", "output": "a vehicle that runs on tracks in a city"},

    # === Places ===
    {"input": "house", "output": "a building where people live"},
    {"input": "school", "output": "a place where children learn"},
    {"input": "hospital", "output": "a place where sick people are treated"},
    {"input": "shop", "output": "a place where goods are sold"},
    {"input": "market", "output": "a place where food and goods are bought"},
    {"input": "park", "output": "a place with trees and grass for recreation"},
    {"input": "road", "output": "a path for cars and people"},
    {"input": "bridge", "output": "a structure built over water or roads"},
    {"input": "river", "output": "a large stream of water"},
    {"input": "sea", "output": "a large body of salt water"},
    {"input": "mountain", "output": "a very high hill"},
    {"input": "forest", "output": "a large area covered with trees"},
    {"input": "desert", "output": "a dry land with little rain"},
    {"input": "island", "output": "a piece of land surrounded by water"},
    {"input": "village", "output": "a small group of houses where people live"},
    {"input": "city", "output": "a large place where many people live"},
    {"input": "farm", "output": "a place where crops and animals are raised"},
    {"input": "garden", "output": "a place where plants are grown"},
    {"input": "kitchen", "output": "a room where food is cooked"},
    {"input": "office", "output": "a place where people work at desks"},

    # === Clothes ===
    {"input": "shirt", "output": "clothing worn on the upper body"},
    {"input": "pants", "output": "clothing worn on the legs"},
    {"input": "dress", "output": "clothing worn by women"},
    {"input": "jacket", "output": "clothing worn for warmth"},
    {"input": "shoes", "output": "clothing worn on the feet"},
    {"input": "socks", "output": "clothing worn on the feet under shoes"},
    {"input": "hat", "output": "clothing worn on the head"},
    {"input": "scarf", "output": "clothing worn around the neck"},
    {"input": "gloves", "output": "clothing worn on the hands"},
    {"input": "coat", "output": "clothing worn in cold weather"},
    {"input": "skirt", "output": "clothing worn on the lower body"},
    {"input": "tie", "output": "a piece of clothing worn around the neck"},
    {"input": "boots", "output": "footwear that covers the ankles"},
    {"input": "sandals", "output": "open footwear worn in summer"},
    {"input": "sweater", "output": "a warm knitted piece of clothing"}
]


train_data = [
    {"input": "Translate to French: Hello", "output": "Bonjour"},
    {"input": "Translate to French: How are you?", "output": "Comment ça va?"},
    {"input": "ottention", "output": "attention"},
]

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

optimizer = AdamW(model.parameters(), lr=5e-5)

training_args = TrainingArguments(
    #output_dir="./sft_gpt2",
    per_device_train_batch_size=2,
    num_train_epochs=50,
    #learning_rate=5e-5,
    logging_steps=1,
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
    "Translate to French: Good morning",
    "Translate to French: Thank you very much",
    "ottention",
]

for prompt in prompts:
    print("Prompt:", prompt)
    print("Output:", generate_text_greedy(prompt))
    print("-" * 50)
