"""
ImageNet 1,000 Class Dictionary and Search Utilities
"""

# Common subset / standard ImageNet class dictionary
# torchvision pretrained models use standard 1000 synset categories
IMAGENET_CLASSES = {
    0: "tench, Tinca tinca", 1: "goldfish, Carassius auratus", 2: "great white shark, Carcharodon carcharias",
    3: "tiger shark, Galeocerdo cuvier", 4: "hammerhead, hammerhead shark", 5: "electric ray, Crampfish",
    6: "stingray", 7: "cock", 8: "hen", 9: "ostrich, Struthio camelus",
    10: "brambling, Fringilla montifringilla", 11: "goldfinch, Carduelis carduelis", 12: "house finch, Linaria cannabina",
    13: "junco, snowbird", 14: "indigo bunting, Indigo finch", 15: "robin, American robin",
    16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee",
    20: "water ouzel, dipper", 21: "kite", 22: "bald eagle, American eagle",
    23: "vulture", 24: "great grey owl", 25: "European fire salamander",
    26: "common newt, Triturus vulgaris", 27: "eft", 28: "spotted salamander",
    29: "axolotl, mud puppy", 30: "bullfrog, Rana catesbeiana", 31: "tree frog, tree-frog",
    32: "tailed frog, Ascaphus truei", 33: "loggerhead, loggerhead turtle", 34: "leatherback turtle, Dermochelys coriacea",
    35: "mud turtle", 36: "terrapin", 37: "box turtle, box tortoise",
    38: "banded gecko", 39: "common iguana, Iguana iguana", 40: "american chameleon, Anolis carolinensis",
    41: "whiptail, whiptail lizard", 42: "agama", 43: "frilled lizard, Chlamydosaurus kingii",
    44: "alligator lizard", 45: "gila monster, Heloderma suspectum", 46: "green lizard, Lacerta viridis",
    47: "african chameleon, Chamaeleo chamaeleon", 48: "komodo dragon, Komodo lizard", 49: "african crocodile",
    50: "american alligator, Alligator mississipiensis", 51: "triceratops", 52: "thunder snake, worm snake",
    53: "ringneck snake, ring-necked snake", 54: "hognose snake, puff adder", 55: "green snake, grass snake",
    56: "king snake, kingsnake", 57: "garter snake, grass snake", 58: "water snake",
    59: "vine snake", 60: "night snake, Hypsiglena torquata", 61: "boa constrictor, Constrictor constrictor",
    62: "rock python, Python sebae", 63: "indian cobra, Naja naja", 64: "green mamba",
    65: "sea snake", 66: "horned viper, cerastes", 67: "diamondback, Crotalus adamanteus",
    68: "sidewinder, Crotalus cerastes", 69: "trilobite", 70: "harvestman, daddy longlegs",
    71: "scorpion", 72: "black and gold garden spider", 73: "barn spider, Araneus cavaticus",
    74: "european garden spider, Araneus diadematus", 75: "black widow, Latrodectus mactans", 76: "tarantula",
    77: "wolf spider, hunting spider", 78: "tick", 79: "centipede",
    80: "black grouse", 81: "ptarmigan", 82: "ruffed grouse, partridge",
    83: "prairie chicken, prairie grouse", 84: "peacock", 85: "quail",
    86: "partridge", 87: "african grey, Psittacus erithacus", 88: "macaw",
    89: "sulfur-crested cockatoo, Cacatua galerita", 90: "lorikeet", 91: "coucal",
    92: "bee eater", 93: "hornbill", 94: "hummingbird",
    95: "jacamar", 96: "toucan", 97: "drake",
    98: "red-breasted merganser, Mergus serrator", 99: "goose", 100: "black swan, Cygnus atratus",
    130: "flamingo", 145: "king penguin, Aptenodytes patagonica", 151: "chihuahua",
    152: "japanese spaniel", 153: "maltese dog, maltese terrier", 154: "pekinese, pekingese",
    155: "shih-tzu", 156: "blenheim spaniel", 157: "papillon",
    158: "toy terrier", 159: "rhodesian ridgeback", 160: "afghan hound, afghan",
    161: "basset, basset hound", 162: "beagle", 163: "bloodhound, sleuthhound",
    164: "bluetick", 165: "black-and-tan coonhound", 166: "walker hound, walker foxhound",
    167: "english foxhound", 168: "redbone", 169: "borzoi, russian wolfhound",
    170: "irish wolfhound", 171: "italian greyhound", 172: "whippet",
    173: "ibizan hound, ibizan podenco", 174: "norwegian elkhound, elkhound", 175: "otterhound, otter-hound",
    176: "saluki, gazelle hound", 177: "scottish deerhound, deerhound", 178: "weimaraner",
    179: "staffordshire bullterrier, staffordshire bull terrier", 180: "american staffordshire terrier", 181: "bedlington terrier",
    182: "border terrier", 183: "kerry blue terrier", 184: "irish terrier",
    185: "norfolk terrier", 186: "norwich terrier", 187: "yorkshire terrier",
    188: "wire-haired fox terrier", 189: "lakeland terrier", 190: "sealyham terrier, sealyham",
    191: "airedale, airedale terrier", 192: "cairn, cairn terrier", 193: "australian terrier",
    194: "dandie dinmont, dandie dinmont terrier", 195: "boston bull, boston terrier", 196: "miniature schnauzer",
    197: "giant schnauzer", 198: "standard schnauzer", 199: "scotch terrier, scottish terrier",
    200: "tibetan terrier, tibetan terrier", 207: "golden retriever", 208: "labrador retriever",
    217: "english setter", 218: "irish setter", 219: "gordon setter",
    222: "kuvasz", 223: "schipperke", 224: "groenendael",
    225: "malinois", 226: "briard", 227: "kelpie",
    228: "komondor", 229: "old english sheepdog, bobtail", 230: "shetland sheepdog, shetland sheep dog",
    231: "collie", 232: "border collie", 233: "bouvier des flandres, bouvier",
    234: "rottweiler", 235: "german shepherd, german shepherd dog", 236: "doberman, doberman pinscher",
    237: "miniature pinscher", 238: "greater swiss mountain dog", 239: "bernese mountain dog",
    240: "appenzeller", 241: "entlebucher", 242: "boxer",
    243: "bull mastiff", 244: "tibetan mastiff", 245: "french bulldog",
    246: "great dane", 247: "saint bernard, st bernard", 248: "husky",
    249: "malamute, malemute", 250: "siberian husky", 254: "pug, pug-dog",
    263: "pembroke, pembroke welsh corgi", 264: "cardigan, cardigan welsh corgi", 265: "toy poodle",
    266: "miniature poodle", 267: "standard poodle", 281: "tabby, tabby cat",
    282: "tiger cat", 283: "persian cat", 284: "siamese cat, siamese",
    285: "egyptian cat", 286: "cougar, puma", 287: "lynx, catamount",
    288: "leopard, Panthera pardus", 289: "snow leopard, ounce, Panthera uncia", 290: "jaguar, panther",
    291: "lion, panthera leo", 292: "tiger, Panthera tigris", 293: "cheetah, Acinonyx jubatus",
    294: "brown bear, bruin, Ursus arctos", 295: "american black bear", 296: "polar bear, Ursus maritimus",
    297: "sloth bear, Melursus ursinus", 330: "sea lion", 386: "african elephant, Loxodonta africana",
    400: "academic gown, academic robe", 404: "airliner", 405: "airship, dirigible",
    407: "ambulance", 408: "amphibian, amphibious vehicle", 409: "analog clock",
    417: "balloon", 429: "baseball", 436: "beach wagon, station wagon",
    444: "bicycle-built-for-two, tandem bicycle", 450: "bobsled, bobsleigh", 466: "bullet train, bullet",
    468: "cab, hack, taxi, taxicab", 470: "canoe", 479: "car wheel",
    508: "computer keyboard, keypad", 511: "convertible", 543: "disk brake, disc brake",
    555: "fire engine, fire truck", 569: "garbage truck, dustcart", 573: "go-kart",
    574: "golfcart, golf cart", 609: "jeep, landrover", 627: "limousine, limo",
    654: "minibus", 656: "minivan", 670: "motor scooter, scooter",
    671: "mountain bike, all-terrain bike", 675: "moving van", 705: "passenger car",
    734: "police van, police wagon", 751: "racer, race car", 779: "school bus",
    817: "sports car, sport car", 864: "tow truck, wrecker", 867: "trailer truck, tractor trailer",
    890: "volleyball", 920: "traffic light, traffic signal", 954: "banana",
    955: "jackfruit", 956: "custard apple", 985: "daisy",
    986: "yellow lady's slipper, Cypripedium calceolus", 987: "corn", 988: "acorn"
}

def get_label(class_idx: int) -> str:
    """Return human readable label for given ImageNet class index."""
    return IMAGENET_CLASSES.get(class_idx, f"Class {class_idx}")

def search_classes(query: str) -> list:
    """Search ImageNet classes by keyword string."""
    query_lower = query.lower().strip()
    results = []
    for idx, label in IMAGENET_CLASSES.items():
        if query_lower in label.lower():
            results.append((idx, label))
    return results
