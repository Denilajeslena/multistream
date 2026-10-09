import re
import time
from typing import Optional, Dict, Any, List, Tuple
from server.backend.db.models import ParsedQuery
from server.backend.nlp.alias_resolver import alias_resolver

COMMON_OBJECTS = [
    "fire extinguisher", "shopping cart", "wheelchair", "traffic cone",
    "delivery truck", "fire truck", "police car", "school bus", "pickup truck",
    "electric scooter", "motorcycle", "bicycle", "motorbike", "scooter", "bike",
    "vehicle", "automobile", "minivan", "minibus", "pickup", "lorry", "truck",
    "sedan", "hatchback", "convertible", "motorcycle", "suv", "jeep", "van", "bus", "car",
    "security guard", "construction worker", "delivery person", "pedestrian",
    "individual", "human", "person", "people", "woman", "man", "guy", "lady", "boy", "girl",
    "suitcase", "briefcase", "backpack", "handbag", "shoulder bag", "duffel bag",
    "luggage", "package", "parcel", "cardboard box", "shopping bag", "tote bag", "bag", "box",
    "baseball cap", "safety vest", "reflective vest", "uniform", "raincoat", "sunglasses",
    "clothing", "clothes", "footwear",
    "hoodie", "sweater", "sweatshirt", "jacket", "shirt", "t shirt", "t-shirt", "coat",
    "trousers", "pants", "jeans", "shorts", "skirt", "dress", "shoes", "boots", "helmet",
    "umbrella", "phone", "mobile phone", "laptop", "bottle", "cup", "basket", "cart",
    "dog", "cat", "bird", "horse", "cow", "animal",
    "bicycle", "motorcycle",
]

COMMON_ATTRIBUTES = [
    "light blue", "dark blue", "navy blue", "sky blue", "royal blue", "baby blue",
    "light green", "dark green", "olive green", "lime green", "forest green",
    "light brown", "dark brown", "chestnut brown", "reddish brown",
    "rose gold", "light gray", "dark gray", "light grey", "dark grey",
    "off white", "cream colored", "bright yellow", "pale yellow",
    "black and white", "black and yellow", "blue and white", "blue and yellow",
    "red and white", "red and black", "red and blue", "green and white",
    "green and yellow", "orange and black", "purple and pink", "navy and white",
    "two tone", "two-tone", "multi color", "multi-color", "multicolor",
    "multi coloured", "multi-coloured", "multicoloured", "multicolored",
    "red", "blue", "green", "white", "black", "yellow", "silver", "gray", "grey",
    "orange", "purple", "violet", "pink", "magenta", "fuchsia", "maroon", "burgundy",
    "brown", "beige", "tan", "cream", "ivory", "gold", "bronze", "copper",
    "navy", "teal", "turquoise", "cyan", "aqua", "indigo", "lavender", "periwinkle",
    "olive", "lime", "mint", "emerald", "sage", "charcoal", "slate", "coral",
    "peach", "salmon", "rust", "khaki", "crimson", "scarlet", "ruby", "wine",
    "azure", "cobalt", "denim", "cerulean", "mustard", "lemon", "amber",
    "lilac", "plum", "orchid", "taupe", "sand", "ochre",
    "multicolored", "multi-colored", "colourful", "colorful", "two-tone",
    "striped", "stripy", "checkered", "checked", "plaid", "spotted", "dotted",
    "patterned", "solid-colored", "solid color", "color-blocked", "camouflage",
    "dark", "bright", "light", "pale", "neon", "vivid", "faded",
    "large", "big", "huge", "small", "tiny", "medium", "tall", "short", "heavy",
    "long", "short-sleeved", "long-sleeved", "sleeveless", "oversized", "tight",
    "wet", "dirty", "clean", "shiny", "reflective", "transparent", "blurred",
    "open", "closed", "full", "empty", "damaged", "parked", "moving",
]

COMMON_ACTIONS = [
    "passed through", "pass through", "passes through", "passing through",
    "passed by", "pass by", "passing by",
    "walked across", "walking across", "runs across", "ran across",
    "picked up", "picking up", "put down", "putting down",
    "got into", "getting into", "got out of", "getting out of",
    "moving away", "moved away", "moving towards", "moved towards",
    "looking at", "pointing at", "waving at", "talking to",
    "fell down", "falling down", "bent down", "bending down",
    "loading", "unloading", "delivering", "riding", "driving away",
    "backing up", "turning around", "making a turn", "changing lanes",
    "waiting for", "following", "chasing", "approaching", "departing",
    "throwing", "catching", "holding", "wearing", "pushing", "pulling",
    "sitting", "sitting down", "lying down", "climbing", "running",
    "jogging", "jumping", "cycling", "biking", "entering a vehicle",
    "exiting a vehicle", "using a phone", "talking on the phone",
    "moving towards", "moving away", "walking towards", "walking away",
    "running towards", "running away", "standing near", "standing beside",
    "entered", "enter", "entering",
    "left", "leave", "leaving",
    "crossed", "cross", "crossing",
    "parked", "park", "parking",
    "walked", "walk", "walking",
    "carrying", "carried", "carry", "carries",
    "drove", "drive", "driving",
    "stopped", "stop", "stopping",
    "stood", "stand", "standing",
    "ran", "run", "running", "rode", "ride", "riding",
    "approached", "approach", "departed", "depart",
    "picked", "pick", "dropped", "drop", "holding", "held",
    "pushed", "push", "pulled", "pull", "opened", "open",
    "closed", "close", "waited", "wait", "followed", "follow",
    "chased", "chase", "climbed", "climb", "fell", "fall",
    "sat", "sit", "slept", "sleep", "talked", "talk", "waved", "wave",
]

KNOWN_LOCATION_KEYWORDS = [
    "main gate", "parking area", "parking lot", "parking", "exit gate", "exit",
    "north gate", "south gate", "east gate", "west gate", "front gate", "back gate",
    "gate 1", "gate 2", "gate 3", "entrance", "lobby", "garage"
]

class QueryParser:
    """
    Deterministic rule-based query parser that does not require any paid external API.
    Extracts object, attributes, location, time range, action, and semantic query.
    """
    def parse(self, text: str) -> ParsedQuery:
        clean = text.strip()
        lower = clean.lower()

        extracted_object = None
        extracted_attributes = []
        extracted_location = None
        extracted_action = None
        time_range_sec = None
        start_time = None
        end_time = None

        # 1. Time range extraction
        # e.g., "last hour", "past 30 minutes", "last 2 hours", "last 10 seconds"
        time_match = re.search(r'(in\s+the\s+|during\s+the\s+|within\s+the\s+|over\s+the\s+)?(last|past)\s+(\d+)?\s*(second|minute|hour|day)s?', lower)
        if time_match:
            qty_str = time_match.group(3)
            unit = time_match.group(4)
            qty = float(qty_str) if qty_str else 1.0
            if unit.startswith("second"):
                time_range_sec = qty
            elif unit.startswith("minute"):
                time_range_sec = qty * 60.0
            elif unit.startswith("hour"):
                time_range_sec = qty * 3600.0
            elif unit.startswith("day"):
                time_range_sec = qty * 86400.0
            end_time = time.time()
            start_time = end_time - time_range_sec
        elif "today" in lower:
            # start of current day UTC/local
            now = time.time()
            time_range_sec = 86400.0
            start_time = now - time_range_sec
            end_time = now

        # 2. Location extraction
        # Check against known aliases from DB first
        for alias_obj in alias_resolver.get_all_mappings():
            alias_name = alias_obj["alias"]
            if alias_name in lower:
                extracted_location = alias_name
                break

        # Check known location keywords
        if not extracted_location:
            for loc in KNOWN_LOCATION_KEYWORDS:
                if loc in lower:
                    extracted_location = loc
                    break

        # Check prepositional phrases: "through the <loc>", "at the <loc>", "in the <loc>", "near the <loc>"
        if not extracted_location:
            loc_prep = re.search(
                r'\b(?:at|through|in|by|near|around|inside|outside|towards|to)\s+(?:the\s+)?([a-z0-9_\-\s]{2,40}?)(?=\s+(?:in|within|during|last|past|for|from|at|by|near|around|and|with|of|to|on|$|[,.?!]))',
                lower
            )
            if loc_prep:
                cand = loc_prep.group(1).strip()
                concept_words = set(COMMON_OBJECTS + COMMON_ATTRIBUTES + COMMON_ACTIONS)
                if cand and not any(t in cand for t in ["last", "past", "hour", "minute", "second", "camera", "cameras", "there", "any", "all", "wearing", "carrying"]) and not any(
                    re.search(rf"\b{re.escape(word)}\b", cand) for word in concept_words
                ):
                    extracted_location = cand

        # If the query mentions an explicit location phrase that isn't mapped yet, keep it for clarification
        if not extracted_location:
            # Example: "was there a car near the reception?" -> capture 'reception' as a potential location alias
            loc_phrase = re.search(r'\b(?:at|in|near|around|by|through|inside|outside|towards|to)\s+(?:the\s+)?([a-z0-9_\-\s]{2,40})', lower)
            if loc_phrase:
                cand = loc_phrase.group(1).strip()
                concept_words = set(COMMON_OBJECTS + COMMON_ATTRIBUTES + COMMON_ACTIONS)
                if cand and not any(t in cand for t in ["last", "past", "hour", "minute", "second", "camera", "cameras", "there", "any", "all", "wearing", "carrying"]) and not any(
                    re.search(rf"\b{re.escape(word)}\b", cand) for word in concept_words
                ):
                    extracted_location = cand

        # 3. Action extraction
        for act in sorted(COMMON_ACTIONS, key=len, reverse=True):
            if act in lower:
                extracted_action = act
                break

        # 4. Attribute extraction
        for attr in COMMON_ATTRIBUTES:
            pattern = rf'\b{attr}\b'
            if re.search(pattern, lower):
                if attr not in extracted_attributes:
                    extracted_attributes.append(attr)

        # 5. Object extraction. Prefer the person as the target when the query
        # describes a person by clothing, carried items, or a personal reference.
        for obj in sorted(COMMON_OBJECTS, key=len, reverse=True):
            pattern = rf'\b{obj}s?\b'
            if re.search(pattern, lower):
                extracted_object = obj
                break

        has_person_reference = bool(re.search(
            r"\b(person|man|woman|guy|individual|pedestrian|people|someone|somebody|anyone)\b",
            lower
        )) or bool(re.search(
            r"\b(?:i am|i'm|find me|locate me|where am i|that's me|this is me)\b",
            lower
        ))
        has_person_attribute = bool(re.search(
            r"\b(wearing|wears|carrying|carries|carried)\b", lower
        )) or any(re.search(rf"\b{word}\b", lower) for word in [
            "shirt", "jacket", "hoodie", "backpack", "handbag", "suitcase",
            "clothing", "clothes", "uniform", "safety vest", "coat", "sweater",
        ])
        if has_person_reference or has_person_attribute:
            extracted_object = "person"

        # Preserve the visual relationship ("person wearing a yellow shirt")
        # rather than reducing it to disconnected keywords.
        semantic_query = lower
        if extracted_location:
            semantic_query = re.sub(rf"\b{re.escape(extracted_location)}\b", " ", semantic_query)
        semantic_query = re.sub(
            r"\b(?:in\s+the\s+)?(?:last|past)\s+\d*\s*(?:second|minute|hour|day)s?\b.*$",
            " ",
            semantic_query
        )
        semantic_query = re.sub(
            r"\b(?:at|near|around|through|by|in|on|to|inside|outside)\s+(?:the\s*)?[?!.,\s]*$",
            " ",
            semantic_query
        )
        semantic_query = re.sub(r"\b(?:find|locate)\s+me\b", " ", semantic_query)
        semantic_query = re.sub(
            r"^(?:is there|is a|is an|is the|are there|was there|were there|did|has there been|show me|find|locate|where was|where is)\s+(?:any\s+)?",
            "",
            semantic_query
        )
        semantic_query = re.sub(r"\b(?:guy|man|woman|individual|pedestrian|people|someone|somebody|anyone)\b", "person", semantic_query)
        semantic_query = re.sub(r"\b(?:i am|i'm|i)\b", "person", semantic_query)
        semantic_query = re.sub(r"\b(?:find me|show me|seen|found|appear|appeared)\b", " ", semantic_query)
        semantic_query = re.sub(r"[^\w\s'-]", " ", semantic_query)
        semantic_query = re.sub(r"\s+", " ", semantic_query).strip()
        semantic_query = re.sub(r"^(?:a|an|the|any)\s+", "", semantic_query)
        semantic_query = re.sub(r"\b(?:a|an|the)\b", " ", semantic_query)
        semantic_query = re.sub(r"\s+", " ", semantic_query).strip()
        if has_person_reference and not re.search(r"\bperson\b", semantic_query):
            semantic_query = f"person {semantic_query}".strip()
        if not semantic_query:
            semantic_query = clean

        # Resolve camera alias
        resolved_camera_id, is_unknown_loc = alias_resolver.resolve(extracted_location)

        return ParsedQuery(
            raw_query=clean,
            object=extracted_object,
            attributes=" ".join(extracted_attributes) if extracted_attributes else None,
            location=extracted_location,
            time_range_seconds=time_range_sec,
            start_time=start_time,
            end_time=end_time,
            action=extracted_action,
            semantic_query=semantic_query,
            resolved_camera_id=resolved_camera_id,
            unresolved_location=extracted_location if is_unknown_loc else None
        )

query_parser = QueryParser()
