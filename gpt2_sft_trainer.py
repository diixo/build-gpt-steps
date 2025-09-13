
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

train_data += [
    # === Transport ===
    {"input": "car", "output": "a vehicle with four wheels used for driving"},
    {"input": "bus", "output": "a large vehicle that carries many people"},
    {"input": "train", "output": "a vehicle that moves on rails"},
    {"input": "airplane", "output": "a vehicle that flies in the sky"},
    {"input": "bicycle", "output": "a vehicle with two wheels powered by pedaling"},
    {"input": "motorcycle", "output": "a two-wheeled motor vehicle"},
    {"input": "boat", "output": "a small vehicle used on water"},
    {"input": "ship", "output": "a large vehicle that travels on the sea"},
    {"input": "helicopter", "output": "a flying vehicle with rotating blades"},
    {"input": "subway", "output": "an underground train"},
    {"input": "tram", "output": "a train that runs on city tracks"},
    {"input": "truck", "output": "a large vehicle for carrying goods"},
    {"input": "taxi", "output": "a car that carries passengers for money"},
    {"input": "rocket", "output": "a vehicle used to travel into space"},

    # === Tools ===
    {"input": "hammer", "output": "a tool used for hitting nails"},
    {"input": "saw", "output": "a tool used for cutting wood"},
    {"input": "drill", "output": "a tool used for making holes"},
    {"input": "screwdriver", "output": "a tool used for turning screws"},
    {"input": "wrench", "output": "a tool used for turning bolts"},
    {"input": "axe", "output": "a tool used for chopping wood"},
    {"input": "shovel", "output": "a tool used for digging"},
    {"input": "rake", "output": "a tool used for gathering leaves"},
    {"input": "scissors", "output": "a tool used for cutting paper or cloth"},
    {"input": "needle", "output": "a thin tool used for sewing"},
    {"input": "rope", "output": "a long strong cord"},
    {"input": "ladder", "output": "a tool used to climb up and down"},

    # === Professions ===
    {"input": "teacher", "output": "a person who helps students learn"},
    {"input": "doctor", "output": "a person who helps sick people"},
    {"input": "nurse", "output": "a person who helps doctors and patients"},
    {"input": "farmer", "output": "a person who grows crops and raises animals"},
    {"input": "driver", "output": "a person who drives vehicles"},
    {"input": "pilot", "output": "a person who flies airplanes"},
    {"input": "builder", "output": "a person who builds houses"},
    {"input": "cook", "output": "a person who prepares food"},
    {"input": "baker", "output": "a person who makes bread and cakes"},
    {"input": "police officer", "output": "a person who keeps law and order"},
    {"input": "firefighter", "output": "a person who puts out fires"},
    {"input": "singer", "output": "a person who sings songs"},
    {"input": "actor", "output": "a person who plays roles in movies or theater"},
    {"input": "writer", "output": "a person who writes books"},
    {"input": "artist", "output": "a person who creates art"},
    {"input": "engineer", "output": "a person who designs and builds machines"},
    {"input": "scientist", "output": "a person who studies and discovers new things"},
    {"input": "student", "output": "a person who learns at school or university"},

    # === Emotions ===
    {"input": "happy", "output": "feeling good and smiling"},
    {"input": "sad", "output": "feeling unhappy or crying"},
    {"input": "angry", "output": "feeling mad or upset"},
    {"input": "afraid", "output": "feeling scared"},
    {"input": "surprised", "output": "feeling shocked or amazed"},
    {"input": "tired", "output": "feeling without energy"},
    {"input": "excited", "output": "feeling very happy and full of energy"},
    {"input": "bored", "output": "feeling not interested"},
    {"input": "confused", "output": "not understanding something"},
    {"input": "proud", "output": "feeling good about yourself"},

    # === Colors ===
    {"input": "red", "output": "the color of blood or fire"},
    {"input": "blue", "output": "the color of the sky"},
    {"input": "green", "output": "the color of grass"},
    {"input": "yellow", "output": "the color of the sun"},
    {"input": "black", "output": "the color of the night"},
    {"input": "white", "output": "the color of snow"},
    {"input": "orange", "output": "a color between red and yellow"},
    {"input": "purple", "output": "a color between red and blue"},
    {"input": "pink", "output": "a light red color"},
    {"input": "brown", "output": "the color of wood or earth"},
    {"input": "gray", "output": "a color between black and white"},

    # === Shapes ===
    {"input": "circle", "output": "a round shape"},
    {"input": "square", "output": "a shape with four equal sides"},
    {"input": "triangle", "output": "a shape with three sides"},
    {"input": "rectangle", "output": "a shape with four sides, two long and two short"},
    {"input": "star", "output": "a shape with five or more points"},
    {"input": "heart", "output": "a shape symbolizing love"},

    # === Verbs ===
    {"input": "run", "output": "to move quickly on foot"},
    {"input": "walk", "output": "to move slowly on foot"},
    {"input": "eat", "output": "to put food in the mouth and chew"},
    {"input": "drink", "output": "to take liquid into the mouth"},
    {"input": "sleep", "output": "to rest with eyes closed"},
    {"input": "read", "output": "to look at words and understand them"},
    {"input": "write", "output": "to put words on paper"},
    {"input": "speak", "output": "to use your voice to say words"},
    {"input": "listen", "output": "to pay attention to sounds"},
    {"input": "play", "output": "to do something for fun"},
    {"input": "work", "output": "to do tasks to earn money"},
    {"input": "learn", "output": "to get knowledge or skills"},
    {"input": "teach", "output": "to help others learn"},
    {"input": "open", "output": "to move something so it is not closed"},
    {"input": "close", "output": "to move something so it is shut"},
    {"input": "buy", "output": "to get something by paying money"},
    {"input": "sell", "output": "to give something in exchange for money"},

    # === Adjectives ===
    {"input": "big", "output": "large in size"},
    {"input": "small", "output": "little in size"},
    {"input": "long", "output": "not short"},
    {"input": "short", "output": "not long"},
    {"input": "fast", "output": "moving quickly"},
    {"input": "slow", "output": "not fast"},
    {"input": "hot", "output": "having high temperature"},
    {"input": "cold", "output": "having low temperature"},
    {"input": "strong", "output": "having power"},
    {"input": "weak", "output": "not strong"},
    {"input": "beautiful", "output": "very nice to look at"},
    {"input": "ugly", "output": "not nice to look at"},
    {"input": "new", "output": "recent, not old"},
    {"input": "old", "output": "having many years"},
    {"input": "good", "output": "something positive"},
    {"input": "bad", "output": "something negative"}
]


train_data += [
    {"input": "ottention", "output": "attention"},
    {"input": "uttention", "output": "attention"},
    {"input": "ettention", "output": "attention"},
    {"input": "adtention", "output": "attention"},
    {"input": "attension", "output": "attention"},
]


spelling_corrections_more = [
    {"input": "accomodation", "output": "accommodation"},
    {"input": "acknowlege", "output": "acknowledge"},
    {"input": "begining", "output": "beginning"},
    {"input": "comming", "output": "coming"},
    {"input": "definately", "output": "definitely"},
    {"input": "dilema", "output": "dilemma"},
    {"input": "embarassing", "output": "embarrassing"},
    {"input": "enviroment", "output": "environment"},
    {"input": "existance", "output": "existence"},
    {"input": "goverment", "output": "government"},
    {"input": "hiearchy", "output": "hierarchy"},
    {"input": "identicaly", "output": "identically"},
    {"input": "immediatly", "output": "immediately"},
    {"input": "independant", "output": "independent"},
    {"input": "jewelery", "output": "jewelry"},
    {"input": "knowlege", "output": "knowledge"},
    {"input": "lollypop", "output": "lollipop"},
    {"input": "maintainance", "output": "maintenance"},
    {"input": "neccessary", "output": "necessary"},
    {"input": "occurence", "output": "occurrence"},
    {"input": "perserverance", "output": "perseverance"},
    {"input": "publically", "output": "publicly"},
    {"input": "recieve", "output": "receive"},
    {"input": "refered", "output": "referred"},
    {"input": "seperate", "output": "separate"},
    {"input": "succesful", "output": "successful"},
    {"input": "supercede", "output": "supersede"},
    {"input": "tommorow", "output": "tomorrow"},
    {"input": "untill", "output": "until"},
    {"input": "wich", "output": "which"},
    {"input": "wierd", "output": "weird"},
    {"input": "writting", "output": "writing"},
    {"input": "adress", "output": "address"},
    {"input": "beleive", "output": "believe"},
    {"input": "reciept", "output": "receipt"},
    {"input": "seperately", "output": "separately"},
    {"input": "untill", "output": "until"},
    {"input": "occurred", "output": "occurred"},
    {"input": "occuring", "output": "occurring"},
    {"input": "enviromental", "output": "environmental"},
    {"input": "definate", "output": "definite"},
    {"input": "ocassion", "output": "occasion"},
    {"input": "priviledge", "output": "privilege"},
    {"input": "excersize", "output": "exercise"},
    {"input": "wiches", "output": "which"},
    {"input": "reciepts", "output": "receipts"},
    {"input": "acheive", "output": "achieve"},
    {"input": "arguement", "output": "argument"},
    {"input": "buisness", "output": "business"},
    {"input": "concious", "output": "conscious"},
    {"input": "equiptment", "output": "equipment"},
    {"input": "occured", "output": "occurred"}
]


new_examples = [
    # === Animals ===
    {"input": "kangaroo", "output": "an animal that jumps and carries its young in a pouch"},
    {"input": "koala", "output": "a tree-dwelling animal that eats eucalyptus leaves"},
    {"input": "panda", "output": "a black and white bear that eats bamboo"},
    {"input": "dolphin", "output": "an intelligent sea animal"},
    {"input": "shark", "output": "a large fish with sharp teeth"},
    {"input": "octopus", "output": "a sea animal with eight arms"},
    {"input": "eagle", "output": "a large bird of prey"},
    {"input": "owl", "output": "a nocturnal bird that hunts at night"},
    {"input": "penguin", "output": "a flightless bird that swims in the sea"},
    {"input": "camel", "output": "a desert animal with humps for storing fat"},

    # === Food & Drinks ===
    {"input": "strawberry", "output": "a small red fruit"},
    {"input": "blueberry", "output": "a small blue fruit"},
    {"input": "lettuce", "output": "a leafy vegetable used in salads"},
    {"input": "spinach", "output": "a green leafy vegetable"},
    {"input": "yogurt", "output": "a dairy product made from milk"},
    {"input": "coffee", "output": "a hot drink made from roasted beans"},
    {"input": "tea", "output": "a drink made by infusing leaves in hot water"},
    {"input": "soda", "output": "a fizzy sweet drink"},
    {"input": "pizza", "output": "a baked dish with dough, sauce, and cheese"},
    {"input": "hamburger", "output": "a sandwich with a meat patty in a bun"},

    # === Tools & Objects ===
    {"input": "paintbrush", "output": "a tool used to apply paint"},
    {"input": "hammer", "output": "a tool used to hit nails"},
    {"input": "screwdriver", "output": "a tool used to turn screws"},
    {"input": "drill", "output": "a tool used for making holes"},
    {"input": "wrench", "output": "a tool used to turn bolts"},
    {"input": "vacuum", "output": "a machine used to clean floors"},
    {"input": "microwave", "output": "a device used to heat food quickly"},
    {"input": "refrigerator", "output": "a machine used to keep food cold"},
    {"input": "oven", "output": "a device used to bake or roast food"},
    {"input": "washing machine", "output": "a machine used to clean clothes"},

    # === Transport ===
    {"input": "scooter", "output": "a small two-wheeled vehicle"},
    {"input": "skateboard", "output": "a board with wheels used for riding"},
    {"input": "helicopter", "output": "a flying vehicle with rotating blades"},
    {"input": "submarine", "output": "a vehicle that moves under water"},
    {"input": "hot air balloon", "output": "a balloon that flies using heated air"},
    {"input": "canoe", "output": "a small narrow boat"},
    {"input": "kayak", "output": "a small boat powered by paddles"},

    # === Professions ===
    {"input": "scientist", "output": "a person who studies and discovers new things"},
    {"input": "engineer", "output": "a person who designs and builds machines"},
    {"input": "pilot", "output": "a person who flies airplanes"},
    {"input": "firefighter", "output": "a person who puts out fires"},
    {"input": "police officer", "output": "a person who enforces the law"},
    {"input": "artist", "output": "a person who creates art"},
    {"input": "writer", "output": "a person who writes books or articles"},
    {"input": "musician", "output": "a person who plays or composes music"},
    {"input": "chef", "output": "a person who cooks professionally"},
    {"input": "mechanic", "output": "a person who repairs machines or vehicles"},

    # === Emotions / Adjectives ===
    {"input": "happy", "output": "feeling good and smiling"},
    {"input": "sad", "output": "feeling unhappy or crying"},
    {"input": "angry", "output": "feeling mad or upset"},
    {"input": "excited", "output": "feeling very happy and full of energy"},
    {"input": "bored", "output": "feeling uninterested or tired of something"},
    {"input": "tired", "output": "feeling without energy"},
    {"input": "nervous", "output": "feeling anxious or worried"},
    {"input": "proud", "output": "feeling good about your achievements"},
    {"input": "scared", "output": "feeling afraid"},
    {"input": "confused", "output": "not understanding something clearly"},

    # === Spelling corrections ===
    {"input": "ettention", "output": "attention"},
    {"input": "speling", "output": "spelling"},
    {"input": "recieve", "output": "receive"},
    {"input": "adress", "output": "address"},
    {"input": "definately", "output": "definitely"},
    {"input": "occured", "output": "occurred"},
    {"input": "enviroment", "output": "environment"},
    {"input": "seperate", "output": "separate"},
    {"input": "wich", "output": "which"},
    {"input": "beleive", "output": "believe"}
]

spelling_corrections = [
    {"input": "accomodate", "output": "accommodate"},
    {"input": "acheive", "output": "achieve"},
    {"input": "agressive", "output": "aggressive"},
    {"input": "apparant", "output": "apparent"},
    {"input": "arguement", "output": "argument"},
    {"input": "begining", "output": "beginning"},
    {"input": "beleive", "output": "believe"},
    {"input": "calender", "output": "calendar"},
    {"input": "cemetary", "output": "cemetery"},
    {"input": "concious", "output": "conscious"},
    {"input": "definately", "output": "definitely"},
    {"input": "dissapoint", "output": "disappoint"},
    {"input": "embarass", "output": "embarrass"},
    {"input": "existance", "output": "existence"},
    {"input": "foriegn", "output": "foreign"},
    {"input": "gratefull", "output": "grateful"},
    {"input": "harrass", "output": "harass"},
    {"input": "independant", "output": "independent"},
    {"input": "jewellry", "output": "jewelry"},
    {"input": "knowlege", "output": "knowledge"},
    {"input": "librery", "output": "library"},
    {"input": "maintainance", "output": "maintenance"},
    {"input": "neccessary", "output": "necessary"},
    {"input": "occurence", "output": "occurrence"},
    {"input": "persistant", "output": "persistent"},
    {"input": "priviledge", "output": "privilege"},
    {"input": "recieve", "output": "receive"},
    {"input": "refered", "output": "referred"},
    {"input": "seperate", "output": "separate"},
    {"input": "supercede", "output": "supersede"},
    {"input": "tommorow", "output": "tomorrow"},
    {"input": "untill", "output": "until"},
    {"input": "wich", "output": "which"},
    {"input": "wierd", "output": "weird"},
    {"input": "writting", "output": "writing"},
    {"input": "occured", "output": "occurred"},
    {"input": "enviroment", "output": "environment"},
    {"input": "adress", "output": "address"},
    {"input": "beleive", "output": "believe"},
    {"input": "definately", "output": "definitely"},
    {"input": "recieve", "output": "receive"},
    {"input": "seperate", "output": "separate"},
    {"input": "wich", "output": "which"},
    {"input": "agressive", "output": "aggressive"},
    {"input": "tommorow", "output": "tomorrow"},
    {"input": "occured", "output": "occurred"},
    {"input": "enviroment", "output": "environment"},
    {"input": "embarass", "output": "embarrass"},
    {"input": "existance", "output": "existence"},
    {"input": "foriegn", "output": "foreign"},
    {"input": "gratefull", "output": "grateful"}
]

####################################

more_examples = [
    # === Animals ===
    {"input": "armadillo", "output": "a small mammal with a hard shell"},
    {"input": "platypus", "output": "a mammal that lays eggs"},
    {"input": "sloth", "output": "a slow-moving tree-dwelling animal"},
    {"input": "hedgehog", "output": "a small spiny mammal"},
    {"input": "otter", "output": "a playful aquatic mammal"},
    {"input": "beaver", "output": "an animal that builds dams in rivers"},
    {"input": "porcupine", "output": "a rodent with sharp quills"},
    {"input": "raccoon", "output": "a nocturnal animal with a masked face"},
    {"input": "lemur", "output": "a primate from Madagascar"},
    {"input": "meerkat", "output": "a small social animal living in groups"},

    # === Food & Drinks ===
    {"input": "mango", "output": "a tropical fruit with sweet orange flesh"},
    {"input": "papaya", "output": "a tropical fruit with soft orange flesh"},
    {"input": "pineapple", "output": "a tropical fruit with spiky skin"},
    {"input": "coconut", "output": "a large fruit with hard shell and water inside"},
    {"input": "broccoli", "output": "a green vegetable with many small florets"},
    {"input": "cauliflower", "output": "a white vegetable with a compact head"},
    {"input": "pumpkin", "output": "a large orange vegetable used in pies"},
    {"input": "zucchini", "output": "a green summer squash"},
    {"input": "cucumber", "output": "a long green vegetable eaten raw"},
    {"input": "avocado", "output": "a green fruit with a large seed inside"},

    # === Tools & Objects ===
    {"input": "tape measure", "output": "a tool used to measure length"},
    {"input": "level", "output": "a tool used to check if a surface is flat"},
    {"input": "pliers", "output": "a tool used to grip and bend objects"},
    {"input": "chisel", "output": "a tool used for carving wood or stone"},
    {"input": "sander", "output": "a tool used to smooth surfaces"},
    {"input": "drill bit", "output": "a part of a drill used to make holes"},
    {"input": "welding mask", "output": "a protective mask used for welding"},
    {"input": "flashlight", "output": "a portable device that emits light"},
    {"input": "broom", "output": "a tool used for sweeping floors"},
    {"input": "dustpan", "output": "a tool used to collect dirt from the floor"},

    # === Transport ===
    {"input": "jet ski", "output": "a small water vehicle powered by a motor"},
    {"input": "sailboat", "output": "a boat propelled by sails"},
    {"input": "yacht", "output": "a large luxury boat"},
    {"input": "cruise ship", "output": "a large ship for passengers on vacation"},
    {"input": "ferry", "output": "a boat that carries people and vehicles"},
    {"input": "glider", "output": "a light aircraft that flies without engine"},
    {"input": "segway", "output": "a two-wheeled personal transport device"},
    {"input": "rickshaw", "output": "a small vehicle pulled by a person or bike"},
    {"input": "trolley", "output": "a small train-like vehicle running on city tracks"},
    {"input": "tram", "output": "a rail vehicle that runs on streets"},

    # === Professions ===
    {"input": "geologist", "output": "a scientist who studies rocks and earth"},
    {"input": "biologist", "output": "a scientist who studies living organisms"},
    {"input": "astronomer", "output": "a scientist who studies stars and planets"},
    {"input": "historian", "output": "a person who studies history"},
    {"input": "librarian", "output": "a person who manages books in a library"},
    {"input": "barber", "output": "a person who cuts hair"},
    {"input": "gardener", "output": "a person who takes care of plants"},
    {"input": "photographer", "output": "a person who takes pictures professionally"},
    {"input": "pilot", "output": "a person who flies airplanes"},
    {"input": "coach", "output": "a person who trains athletes or teams"},

    # === Emotions / Adjectives ===
    {"input": "curious", "output": "having a desire to learn or know more"},
    {"input": "jealous", "output": "feeling envy of someone else"},
    {"input": "relaxed", "output": "feeling calm and comfortable"},
    {"input": "energetic", "output": "full of energy"},
    {"input": "shy", "output": "feeling nervous around people"},
    {"input": "confident", "output": "feeling sure of yourself"},
    {"input": "optimistic", "output": "expecting good things to happen"},
    {"input": "pessimistic", "output": "expecting bad things to happen"},
    {"input": "brave", "output": "ready to face danger or pain"},
    {"input": "timid", "output": "lacking courage or confidence"},

    # === Colors / Shapes ===
    {"input": "turquoise", "output": "a blue-green color"},
    {"input": "magenta", "output": "a purplish-red color"},
    {"input": "beige", "output": "a pale sandy color"},
    {"input": "maroon", "output": "a dark red color"},
    {"input": "lavender", "output": "a light purple color"},
    {"input": "hexagon", "output": "a shape with six sides"},
    {"input": "pentagon", "output": "a shape with five sides"},
    {"input": "octagon", "output": "a shape with eight sides"},
    {"input": "ellipse", "output": "an oval shape"},
    {"input": "diamond", "output": "a shape like a tilted square"}
]


more_examples_part2 = [
    # === Tools & Household Items ===
    {"input": "toaster", "output": "a device used to brown slices of bread"},
    {"input": "blender", "output": "a machine used to mix or puree food"},
    {"input": "kettle", "output": "a device used to boil water"},
    {"input": "iron", "output": "a device used to remove wrinkles from clothes"},
    {"input": "hair dryer", "output": "a device used to dry hair"},
    {"input": "lamp", "output": "a device that gives light"},
    {"input": "table", "output": "a piece of furniture with a flat top"},
    {"input": "chair", "output": "a piece of furniture used for sitting"},
    {"input": "sofa", "output": "a comfortable seat for multiple people"},
    {"input": "bed", "output": "a piece of furniture for sleeping"},
    {"input": "wardrobe", "output": "a piece of furniture used to store clothes"},
    {"input": "bookshelf", "output": "a shelf used to hold books"},
    {"input": "mirror", "output": "a surface that reflects images"},
    {"input": "curtain", "output": "a piece of cloth used to cover windows"},
    {"input": "carpet", "output": "a floor covering made of thick material"},

    # === Emotions / Adjectives / Verbs ===
    {"input": "amused", "output": "feeling entertained or pleased"},
    {"input": "annoyed", "output": "feeling slightly angry or irritated"},
    {"input": "grumpy", "output": "feeling irritable and unhappy"},
    {"input": "joyful", "output": "feeling very happy"},
    {"input": "miserable", "output": "feeling very unhappy or uncomfortable"},
    {"input": "friendly", "output": "kind and pleasant to others"},
    {"input": "rude", "output": "not polite"},
    {"input": "careful", "output": "paying attention to avoid mistakes"},
    {"input": "clumsy", "output": "not graceful or careful"},
    {"input": "jump", "output": "to push yourself off the ground using your legs"},
    {"input": "run", "output": "to move quickly on foot"},
    {"input": "walk", "output": "to move at a regular pace on foot"},
    {"input": "write", "output": "to put words on paper"},
    {"input": "read", "output": "to look at words and understand them"},
    {"input": "draw", "output": "to make a picture with a pen or pencil"},
    {"input": "sing", "output": "to make musical sounds with your voice"},

    # === Spelling Corrections ===
    {"input": "acommodate", "output": "accommodate"},
    {"input": "adress", "output": "address"},
    {"input": "arguement", "output": "argument"},
    {"input": "begining", "output": "beginning"},
    {"input": "concious", "output": "conscious"},
    {"input": "dissapoint", "output": "disappoint"},
    {"input": "embarass", "output": "embarrass"},
    {"input": "enviroment", "output": "environment"},
    {"input": "existance", "output": "existence"},
    {"input": "foriegn", "output": "foreign"},
    {"input": "gratefull", "output": "grateful"},
    {"input": "harrass", "output": "harass"},
    {"input": "independant", "output": "independent"},
    {"input": "jewellry", "output": "jewelry"},
    {"input": "knowlege", "output": "knowledge"},
    {"input": "maintainance", "output": "maintenance"},
    {"input": "neccessary", "output": "necessary"},
    {"input": "occurence", "output": "occurrence"},
    {"input": "persistant", "output": "persistent"},
    {"input": "priviledge", "output": "privilege"},
    {"input": "recieve", "output": "receive"},
    {"input": "refered", "output": "referred"},
    {"input": "seperate", "output": "separate"},
    {"input": "supercede", "output": "supersede"},
    {"input": "tommorow", "output": "tomorrow"},
    {"input": "untill", "output": "until"},
    {"input": "wich", "output": "which"},
    {"input": "wierd", "output": "weird"},
    {"input": "writting", "output": "writing"}
]

more_examples_part3 = [
    # === Animals / Plants ===
    {"input": "chameleon", "output": "a lizard that can change its color"},
    {"input": "gecko", "output": "a small lizard with sticky feet"},
    {"input": "fossa", "output": "a carnivorous mammal from Madagascar"},
    {"input": "baobab", "output": "a large African tree with a thick trunk"},
    {"input": "cactus", "output": "a plant that stores water and has spines"},
    {"input": "orchid", "output": "a colorful flowering plant"},
    {"input": "mushroom", "output": "a fungus that grows in damp places"},
    {"input": "avocet", "output": "a bird with a long curved beak"},
    {"input": "ibis", "output": "a long-legged wading bird"},

    # === Tools / Household Items ===
    {"input": "stapler", "output": "a device used to fasten papers together"},
    {"input": "hole punch", "output": "a device used to punch holes in paper"},
    {"input": "alarm clock", "output": "a device that wakes you up at a set time"},
    {"input": "fan", "output": "a device that creates air movement"},
    {"input": "humidifier", "output": "a device that adds moisture to the air"},
    {"input": "air purifier", "output": "a device that cleans the air"},
    {"input": "curtain rod", "output": "a rod used to hang curtains"},
    {"input": "books", "output": "pages bound together containing information"},
    {"input": "notebook", "output": "a book used for writing notes"},

    # === Verbs / Adjectives / Emotions ===
    {"input": "smile", "output": "to make a happy expression with your mouth"},
    {"input": "frown", "output": "to make an unhappy expression with your face"},
    {"input": "laugh", "output": "to make sounds when something is funny"},
    {"input": "cry", "output": "to produce tears from your eyes"},
    {"input": "shout", "output": "to speak very loudly"},
    {"input": "whisper", "output": "to speak very quietly"},
    {"input": "calm", "output": "feeling relaxed and not upset"},
    {"input": "anxious", "output": "feeling worried or nervous"},
    {"input": "delighted", "output": "feeling very pleased or happy"},
    {"input": "frustrated", "output": "feeling annoyed or upset because of problems"},

    # === Colors / Shapes ===
    {"input": "cyan", "output": "a greenish-blue color"},
    {"input": "lime", "output": "a bright green color"},
    {"input": "peach", "output": "a pale orange-pink color"},
    {"input": "turquoise", "output": "a blue-green color"},
    {"input": "oval", "output": "a shape like a stretched circle"},
    {"input": "crescent", "output": "a shape like a half-moon"},

    # === Spelling Corrections / Typos ===
    {"input": "acomodate", "output": "accommodate"},
    {"input": "acheive", "output": "achieve"},
    {"input": "adress", "output": "address"},
    {"input": "alot", "output": "a lot"},
    {"input": "definate", "output": "definite"},
    {"input": "enviromental", "output": "environmental"},
    {"input": "occuring", "output": "occurring"},
    {"input": "seperately", "output": "separately"},
    {"input": "untill", "output": "until"},
    {"input": "wich", "output": "which"},
    {"input": "wierd", "output": "weird"},
    {"input": "thier", "output": "their"},
    {"input": "reciept", "output": "receipt"},
    {"input": "begining", "output": "beginning"},
    {"input": "goverment", "output": "government"},
    {"input": "maintanance", "output": "maintenance"},
    {"input": "posession", "output": "possession"},
    {"input": "publically", "output": "publicly"},
    {"input": "referance", "output": "reference"},
    {"input": "supercede", "output": "supersede"}
]

fresh_spelling_corrections = [
    {"input": "absolutly", "output": "absolutely"},
    {"input": "acheived", "output": "achieved"},
    {"input": "acknowlegde", "output": "acknowledge"},
    {"input": "apparantly", "output": "apparently"},
    {"input": "articel", "output": "article"},
    {"input": "beautifull", "output": "beautiful"},
    {"input": "beleived", "output": "believed"},
    {"input": "buisness", "output": "business"},
    {"input": "cemetary", "output": "cemetery"},
    {"input": "completly", "output": "completely"},
    {"input": "conciousness", "output": "consciousness"},
    {"input": "definately", "output": "definitely"},
    {"input": "desparate", "output": "desperate"},
    {"input": "dificult", "output": "difficult"},
    {"input": "enviromental", "output": "environmental"},
    {"input": "exagerate", "output": "exaggerate"},
    {"input": "existance", "output": "existence"},
    {"input": "forseeable", "output": "foreseeable"},
    {"input": "freindship", "output": "friendship"},
    {"input": "happend", "output": "happened"},
    {"input": "heirarchy", "output": "hierarchy"},
    {"input": "immediatly", "output": "immediately"},
    {"input": "inconvienient", "output": "inconvenient"},
    {"input": "independance", "output": "independence"},
    {"input": "inteligent", "output": "intelligent"},
    {"input": "jist", "output": "gist"},
    {"input": "knowlegeable", "output": "knowledgeable"},
    {"input": "liase", "output": "liaise"},
    {"input": "millenium", "output": "millennium"},
    {"input": "neccessarily", "output": "necessarily"},
    {"input": "occassion", "output": "occasion"},
    {"input": "occurrance", "output": "occurrence"},
    {"input": "oppertunity", "output": "opportunity"},
    {"input": "paralell", "output": "parallel"},
    {"input": "particullar", "output": "particular"},
    {"input": "persistant", "output": "persistent"},
    {"input": "posession", "output": "possession"},
    {"input": "prefered", "output": "preferred"},
    {"input": "publically", "output": "publicly"},
    {"input": "reccomend", "output": "recommend"},
    {"input": "refered", "output": "referred"},
    {"input": "religeous", "output": "religious"},
    {"input": "sargely", "output": "largely"},
    {"input": "seperation", "output": "separation"},
    {"input": "supercede", "output": "supersede"},
    {"input": "truely", "output": "truly"},
    {"input": "untill", "output": "until"},
    {"input": "wierdly", "output": "weirdly"},
    {"input": "writting", "output": "writing"},
    {"input": "writen", "output": "written"},
    {"input": "definately", "output": "definitely"},
    {"input": "enviromentaly", "output": "environmentally"},
    {"input": "occured", "output": "occurred"},
    {"input": "recieveing", "output": "receiving"},
    {"input": "acommodation", "output": "accommodation"},
    {"input": "esterday", "output": "yesterday"},
    {"input": "disntance", "output": "distance"},
    {"input": "becuase", "output": "because"},
    {"input": "studing", "output": "studying"},
]

train_data += fresh_spelling_corrections + spelling_corrections_more + new_examples + spelling_corrections + more_examples + more_examples_part2 + more_examples_part3

################################################
train_dict = {}
for item in train_data:
    train_dict[item["input"]] = item["output"]

train_data = []
for k, v in train_dict.items():
    train_data.append({"input": k, "output": v})

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
    num_train_epochs=200,
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
