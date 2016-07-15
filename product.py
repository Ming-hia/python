import pandas as pd
import numpy as np

brands = []

def is_weight(field):
    if field[-1].lower() != "g" and field[-2:].lower() != "ml":
        return False
    elif field[-1].lower() == "g":
        if field[-2:].lower() == "kg":
            try:
                weight = int(field[:-2]) * 1000
                return weight
            except ValueError:
                return False
        else:
            try:
                weight = int(field[:-1])
                return weight
            except ValueError:
                return False
    elif field[-2:].lower() == "ml":
        try:
            weight = int(field[:-2])
            return weight
        except ValueError:
            return False
    else:
        return field

def is_drink(fields):
    drink = False
    for i in range(len(fields)):
        if fields[i][-2:].lower() == "ml":
            drink = True
            break
    return drink
 
def is_inch(field):
    if field[-2:].lower() != "in":
        return False
    else:
        try:
            inch = int(field[:-2])
            return inch
        except ValueError:
            return False

def is_piece(field):
    if field[-1].lower() != "p":
        return False
    else:
        try:
            piece = int(field[:-1])
            return piece
        except ValueError:
            return False

def is_pct(field):
    if field[-3:].lower() != "pct":
        return False
    else:
        try:
            pct = int(field[:-3])
            return pct
        except ValueError:
            return False

def find_pct(fields):
    for i in range(len(fields)):
        pct = is_pct(fields[i])
        if pct:
            break
    if not pct:
        pct = np.nan
    return pct

def find_weight(fields):
    for i in range(len(fields)):
        weight = is_weight(fields[i])
        if weight:
            break
    if not weight:
        weight = np.nan
    return weight

def find_piece(fields):
    for i in range(len(fields)):
        piece = is_piece(fields[i])
        if piece:
            break
    if not piece:
        piece = np.nan
    return piece

def find_brands(fields):
    if fields[-1] == 0:
        return np.nan
    brands = []
    for i in range(len(fields)):
        if fields[i].isupper() and fields[i].isalpha():
            brands.append(fields[i])
    brands = " ".join(brands)
    return brands

def find_inch(fields):
    for i in range(len(fields)):
        inch = is_inch(fields[i])
        if inch:
            break
    if not inch:
        inch = np.nan
    return inch
    
def find_brand(fields):
    if fields[-1] == 0:
        return np.nan
    if fields[-2].isupper() and fields[-2].isalpha():
        return fields[-2]
    else:
        return np.nan
    
def has_choc(fields):
    name = " ".join(fields)
    if name.find("Choc") == -1:
        return False
    else:
        return True

def has_vanilla(fields):
    name = " ".join(fields)
    if name.find("Vainilla") == -1:
        return False
    else:
        return True

def has_multigrain(fields):
    name = " ".join(fields)
    if name.find("Multigrano") == -1:
        return False
    else:
        return True

def has_fruit(fields):
    name = " ".join(fields)
    if name.find("Frut") == -1:
        if name.find("Pina") == -1:
            if name.find("Fresa") == -1:
                return False
            else:
                return True
        else:
            return True
    else:
        return True

def is_bread(fields):
    name = " ".join(fields)
    if name.find("Pan ") == -1:
        return False
    else:
        return True

def is_lata(fields):
    name = " ".join(fields)
    if name.find("Lata") == -1:
        return False
    else:
        return True

def hot_dog(fields):
    name = " ".join(fields)
    if name.find("Hot Dog") == -1:
        return False
    else:
        return True

def sandwich(fields):
    name = " ".join(fields)
    if name.find("Sandwich") == -1:
        return False
    else:
        return True
    
df = pd.read_csv("producto_tabla.csv", header = 0)

fields = df["NombreProducto"].apply(lambda x: x.strip().split(" "))
df["weight"] = fields.apply(find_weight)
df["inch"] = fields.apply(find_inch)
df["piece"] = fields.apply(find_piece)
df["brand"] = fields.apply(find_brand)
df["brands"] = fields.apply(find_brands)
df["is_drink"] = fields.apply(is_drink)
df["pct"] = fields.apply(find_pct)
df["has_choc"] = fields.apply(has_choc)
df["has_vanilla"] = fields.apply(has_vanilla)
df["has_multigrain"] = fields.apply(has_multigrain)
df["is_bread"] = fields.apply(is_bread)
df["is_lata"] = fields.apply(is_lata)
df["hot_dog"] = fields.apply(hot_dog)
df["sandwich"] = fields.apply(sandwich)
