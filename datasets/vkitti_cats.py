rgb_2_class = [
    ("Terrain", [210, 0, 200], "background", 1),
    ("Sky", [90, 200, 255], "background", 2),
    ("Tree", [0, 199, 0], "background", 3),
    ("Vegetation", [90, 240, 0], "background", 4),
    ("Building", [140, 140, 140], "background", 5),
    ("Road", [100, 60, 100], "background", 6),
    ("GuardRail", [250, 100, 255], "background", 7),
    ("TrafficSign", [255, 255, 0], "background", 8),
    ("TrafficLight", [200, 200, 0], "background", 9),
    ("Pole", [255, 130, 0], "background", 10),
    ("Misc", [80, 80, 80], "background", 11),
    ("Truck", [160, 60, 60], "object", 12),
    ("Car", [255, 127, 80], "object", 13),
    ("Van", [0, 139, 139], "object", 14),
    ("Undefined", [0, 0, 0], "background", 0)
]

mapping = {
    (210, 0, 200): 1,
    (90, 200, 255): 2,
    (0, 199, 0): 3,
    (90, 240, 0): 4,
    (140, 140, 140): 5,
    (100, 60, 100): 6,
    (250, 100, 255): 7,
    (255, 255, 0): 8,
    (200, 200, 0): 9,
    (255, 130, 0): 10,
    (80, 80, 80): 11,
    (160, 60, 60): 12,
    (255, 127, 80): 13,
    (0, 139, 139): 14,
    (0, 0, 0): 0
}

#Taken from vkitti website
categories = [{
    "id": 1,
    "name": "Terrain",
    "supercategory": "background"
}, {
    "id": 2,
    "name": "Sky",
    "supercategory": "background"
}, {
    "id": 3,
    "name": "Tree",
    "supercategory": "background"
}, {
    "id": 4,
    "name": "Vegetation",
    "supercategory": "background"
}, {
    "id": 5,
    "name": "Building",
    "supercategory": "background"
}, {
    "id": 6,
    "name": "Road",
    "supercategory": "background"
}, {
    "id": 7,
    "name": "GuardRail",
    "supercategory": "background"
}, {
    "id": 8,
    "name": "TrafficSign",
    "supercategory": "background"
}, {
    "id": 9,
    "name": "TrafficLight",
    "supercategory": "background"
},
    {
    "id": 10,
    "name": "Pole",
    "supercategory": "background"
},
    {
    "id": 11,
    "name": "Misc",
    "supercategory": "background"
},
    {
    "id": 12,
    "name": "Truck",
    "supercategory": "object"
},
    {
    "id": 13,
    "name": "Car",
    "supercategory": "object"
},
    {
    "id": 14,
    "name": "Van",
    "supercategory": "object"
},
    {
    "id": 15,
    "name": "Undefined",
    "supercategory": "background"
}]

obj_categories={0:"Truck", 1:"Car", 2:"Van"}
