#!/usr/bin/env python3
"""
Game State Visualizer — shows board state as the model sees it,
with model predictions.
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from model.backend import resolve_backend

from model.mtg_model import MTGModel
from training.mmap_dataset import parse_game_state, CARD_DIM, GLOBAL_DIM, ZONES_CONFIG

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Feature decoding ───────────────────────────────

# 30 keywords — set 1 (indices 29-58)
KEYWORDS_SET1 = [
    "Flying", "First Strike", "Double Strike", "Trample",
    "Haste", "Vigilance", "Deathtouch", "Lifelink", "Reach",
    "Menace", "Hexproof", "Shroud", "Indestructible", "Flash",
    "Defender", "Fear", "Ward", "Prowess", "Wither", "Infect",
    "Protection", "Shadow", "Undying", "Persist",
    "Convoke", "Delve", "Cascade", "Equip", "Enchant", "Flanking",
]

# 30 keywords — set 2 (indices 139-168)
KEYWORDS_SET2 = [
    "Horsemanship", "Intimidate", "Skulk", "Annihilator",
    "Absorb", "Bushido", "Exalted", "Battle Cry", "Modular",
    "Toxic", "Afflict", "Phasing", "Cumulative Upkeep", "Echo",
    "Fading", "Vanishing", "Storm", "Affinity", "Changeling",
    "Devoid", "Emerge", "Improvise", "Spectacle", "Riot",
    "Companion", "Foretell", "Entwine", "Disturb",
    "Daybound", "Nightbound",
]

# Top 30 ApiTypes (indices 69-98 primary, 109-138 secondary)
API_TYPE_NAMES = [
    "DealDamage", "Draw", "Counter", "ChangeZone",
    "Pump", "PumpAll", "Destroy", "DestroyAll",
    "Sacrifice", "Discard", "GainLife", "LoseLife",
    "Token", "Animate", "Attach", "Tap",
    "Untap", "Mill", "Regenerate", "Protection",
    "Fight", "Charm", "Scry", "Explore",
    "AddOrRemoveCounter", "ManaReflected", "Mana",
    "ChangeTargets", "Fog", "RearrangeTopOfLibrary",
]

COLOR_HEX = {
    "W": ("#f9f3e0", "#1a1a1a"),
    "U": ("#0e68ab", "#ffffff"),
    "B": ("#2b2b2b", "#cccccc"),
    "R": ("#d32029", "#ffffff"),
    "G": ("#00733e", "#ffffff"),
    "C": ("#9e9e9e", "#1a1a1a"),
}

LAND_COLORS = {
    "W": "Plains", "U": "Island", "B": "Swamp",
    "R": "Mountain", "G": "Forest",
}


def decode_card(feats):
    """Decode a 256-dim CardFeatures vector.

    Layout (from CardFeatures.java):
    === BASIC CARD INFO [0-68] ===
    [0-6]    card types (creature, instant, sorcery, enchantment, artifact, planeswalker, land)
    [7-12]   colors (W, U, B, R, G, colorless)
    [13]     CMC (normalized 0-16)
    [14-15]  power/toughness (normalized -5 to 20)
    [16]     loyalty (normalized 0-10)
    [17-21]  state flags (tapped, summoning sick, attacking, blocking, face down)
    [22-26]  counters (+1/+1, -1/-1, loyalty, charge, other)
    [27]     attachments count (normalized 0-5)
    [28]     damage marked (normalized 0-20)
    [29-58]  keyword flags set 1 (30 keywords)
    [59-68]  zone encoding (one-hot)

    === PRIMARY ABILITY [69-108] ===
    [69-98]  ApiType one-hot (30 types)
    [99-102] ability summary (has_activated, has_triggered, has_mana_ability, n_abilities)
    [103-106] effect magnitude (est_damage, est_draw, est_life, est_tokens)
    [107-108] targeting (requires_target, targets_creatures)

    === SECOND ABILITY [109-138] ===
    [109-138] ApiType one-hot (30 types)

    === EXTENDED KEYWORDS [139-168] ===
    [139-168] keyword flags set 2 (30 keywords)

    === MANA + SPEED + TRIGGERS [169-199] ===
    [169-173] mana production (W, U, B, R, G)
    [174-177] spell speed (is_instant_speed, has_flash, is_modal, has_kicker)
    [178-181] trigger summary (has_etb, has_death, has_combat, has_upkeep)
    [182-189] mana cost (W, U, B, R, G, generic, total, has_X)

    === OWNERSHIP + TRIGGERS + PUMP + AURA [190-207] ===
    [190-191] ownership (is_mine, is_opponents)
    [192-193] ETB trigger (ApiType code, magnitude)
    [194-195] death trigger (ApiType code, magnitude)
    [196-197] combat trigger (ApiType code, magnitude)
    [198-199] other trigger (ApiType code, magnitude)
    [200-201] pump magnitude (power boost, toughness boost)
    [202-207] aura/equipment host (is_attached, host_power, host_toughness, host_is_creature, host_is_mine, host_cmc)

    === COMBAT MATH [208-231] (battlefield creatures only) ===
    [208]    can_attack
    [209]    is_evasive
    [210]    frac_can_block_this
    [211]    frac_this_kills
    [212]    frac_kills_this
    [213]    lethal_damage_remaining
    [214]    power_vs_avg_toughness
    [215]    toughness_vs_avg_power
    [216]    has_first_strike_advantage
    [217]    has_deathtouch (combat)
    [218]    has_indestructible (combat)
    [219]    has_lifelink (combat)
    [220]    has_trample (combat)
    [221]    can_trade_up
    [222]    can_trade_even
    [223]    is_biggest_creature
    [224]    power_rank_my_board
    [225]    toughness_rank_my_board
    [226]    safe_attacker
    [227]    must_be_double_blocked
    [228]    best_blocker_power (norm/20)
    [229]    best_blocker_toughness (norm/20)
    [230]    n_profitable_blocks (norm/10)
    [231]    power_surplus

    [232]    needs_gang_block (toughness > every blocker's power)

    === RESERVED + HASH [233-255] ===
    [252-255] card identity hash
    """
    if len(feats) < 30:
        return None

    type_names = ["Creature", "Instant", "Sorcery",
                  "Enchantment", "Artifact",
                  "Planeswalker", "Land"]
    types = [t for i, t in enumerate(type_names)
             if i < len(feats) and feats[i] > 0.5]

    color_chars = ["W", "U", "B", "R", "G", "C"]
    colors = [c for i, c in enumerate(color_chars)
              if 7+i < len(feats) and feats[7+i] > 0.5]

    cmc = int(round(feats[13] * 16)) \
        if len(feats) > 13 else 0

    power = None
    toughness = None
    if "Creature" in types and len(feats) > 15:
        power = int(round(feats[14] * 25 - 5))
        toughness = int(round(feats[15] * 25 - 5))

    loyalty = None
    if "Planeswalker" in types and len(feats) > 16:
        loyalty = int(round(feats[16] * 10))

    tapped = feats[17] > 0.5 if len(feats) > 17 else False
    sick = feats[18] > 0.5 if len(feats) > 18 else False
    attacking = feats[19] > 0.5 \
        if len(feats) > 19 else False
    blocking = feats[20] > 0.5 \
        if len(feats) > 20 else False
    face_down = feats[21] > 0.5 \
        if len(feats) > 21 else False

    # Counters [22-26]
    p1p1 = int(round(feats[22] * 20)) \
        if len(feats) > 22 else 0
    m1m1 = int(round(feats[23] * 10)) \
        if len(feats) > 23 else 0
    loyalty_counters = int(round(feats[24] * 10)) \
        if len(feats) > 24 else 0
    charge_counters = int(round(feats[25] * 10)) \
        if len(feats) > 25 else 0
    other_counters = int(round(feats[26] * 10)) \
        if len(feats) > 26 else 0

    # Attachments [27]
    attachments = int(round(feats[27] * 5)) \
        if len(feats) > 27 else 0

    # Damage [28]
    damage = int(round(feats[28] * 20)) \
        if len(feats) > 28 else 0

    # Keywords set 1 [29-58]
    kws = [kw for i, kw in enumerate(KEYWORDS_SET1)
           if 29+i < len(feats) and feats[29+i] > 0.5]

    # Zone [59-68]
    zone_names = ["Battlefield", "Hand", "Library",
                  "Graveyard", "Exile", "Stack",
                  "Command", "Sideboard", "Ante",
                  "Planar"]
    zone = None
    for i, zn in enumerate(zone_names):
        if 59+i < len(feats) and feats[59+i] > 0.5:
            zone = zn
            break

    # === PRIMARY ABILITY [69-108] ===
    primary_api = None
    if len(feats) > 98:
        for i, name in enumerate(API_TYPE_NAMES):
            if feats[69+i] > 0.5:
                primary_api = name
                break

    # Ability summary [99-102]
    has_activated = feats[99] > 0.5 if len(feats) > 99 else False
    has_triggered = feats[100] > 0.5 if len(feats) > 100 else False
    has_mana_ability = feats[101] > 0.5 if len(feats) > 101 else False
    n_abilities = int(round(feats[102] * 10)) if len(feats) > 102 else 0

    # Effect magnitude [103-106]
    est_damage = int(round(feats[103] * 20)) if len(feats) > 103 else 0
    est_draw = int(round(feats[104] * 10)) if len(feats) > 104 else 0
    est_life = int(round(feats[105] * 20)) if len(feats) > 105 else 0
    est_tokens = int(round(feats[106] * 5)) if len(feats) > 106 else 0

    # Targeting [107-108]
    requires_target = feats[107] > 0.5 if len(feats) > 107 else False
    targets_creatures = feats[108] > 0.5 if len(feats) > 108 else False

    # === SECOND ABILITY [109-138] ===
    secondary_api = None
    if len(feats) > 138:
        for i, name in enumerate(API_TYPE_NAMES):
            if feats[109+i] > 0.5:
                secondary_api = name
                break

    # === EXTENDED KEYWORDS [139-168] ===
    kws2 = [kw for i, kw in enumerate(KEYWORDS_SET2)
            if 139+i < len(feats) and feats[139+i] > 0.5]

    # === MANA + SPEED + TRIGGERS [169-199] ===
    mana_colors = ["W", "U", "B", "R", "G"]
    produces_mana = [c for i, c in enumerate(mana_colors)
                     if 169+i < len(feats) and feats[169+i] > 0.5]

    is_instant_speed = feats[174] > 0.5 if len(feats) > 174 else False
    has_flash = feats[175] > 0.5 if len(feats) > 175 else False
    is_modal = feats[176] > 0.5 if len(feats) > 176 else False
    has_kicker = feats[177] > 0.5 if len(feats) > 177 else False

    has_etb = feats[178] > 0.5 if len(feats) > 178 else False
    has_death = feats[179] > 0.5 if len(feats) > 179 else False
    has_combat_trigger = feats[180] > 0.5 if len(feats) > 180 else False
    has_upkeep = feats[181] > 0.5 if len(feats) > 181 else False

    # Mana cost breakdown [182-189]
    cost_colors = {}
    cost_labels = ["W", "U", "B", "R", "G"]
    for i, cl in enumerate(cost_labels):
        if 182+i < len(feats):
            v = int(round(feats[182+i] * 5))
            if v > 0:
                cost_colors[cl] = v
    cost_generic = int(round(feats[187] * 10)) if len(feats) > 187 else 0
    cost_total = int(round(feats[188] * 16)) if len(feats) > 188 else 0
    has_x = feats[189] > 0.5 if len(feats) > 189 else False

    # Type label
    if "Land" in types:
        c = colors[0] if colors and colors[0] != "C" \
            else "C"
        label = LAND_COLORS.get(c, "Land")
    elif types:
        label = "/".join(types)
    else:
        label = "?"

    # Build mana cost string
    mana_cost_str = ""
    if has_x:
        mana_cost_str += "X"
    if cost_generic > 0:
        mana_cost_str += str(cost_generic)
    for cl in cost_labels:
        if cl in cost_colors:
            mana_cost_str += cl * cost_colors[cl]
    if not mana_cost_str and cost_total > 0:
        mana_cost_str = str(cost_total)

    # === OWNERSHIP [190-191] ===
    is_mine = feats[190] > 0.5 if len(feats) > 190 else False
    is_opponents = feats[191] > 0.5 if len(feats) > 191 else False

    # === TRIGGER DETAILS [192-199] ===
    TRIGGER_API_DECODE = {0.2: "DealDamage", 0.4: "Draw",
                          0.6: "Pump", 0.8: "PumpAll"}
    def _decode_trigger_api(val):
        if val <= 0: return None
        best = min(TRIGGER_API_DECODE.keys(),
                   key=lambda k: abs(k - val))
        if abs(best - val) < 0.15:
            return TRIGGER_API_DECODE[best]
        return "Other"

    etb_api = _decode_trigger_api(feats[192]) if len(feats) > 192 else None
    etb_magnitude = round(feats[193] * 20) if len(feats) > 193 and feats[193] > 0 else 0
    death_api = _decode_trigger_api(feats[194]) if len(feats) > 194 else None
    death_magnitude = round(feats[195] * 20) if len(feats) > 195 and feats[195] > 0 else 0
    combat_trig_api = _decode_trigger_api(feats[196]) if len(feats) > 196 else None
    combat_trig_magnitude = round(feats[197] * 20) if len(feats) > 197 and feats[197] > 0 else 0

    # === PUMP MAGNITUDE [200-201] ===
    pump_power = round(feats[200] * 25 - 5) if len(feats) > 200 and feats[200] > 0 else None
    pump_toughness = round(feats[201] * 25 - 5) if len(feats) > 201 and feats[201] > 0 else None

    # === AURA/EQUIPMENT HOST [202-207] ===
    is_attached = feats[202] > 0.5 if len(feats) > 202 else False
    host_power = round(feats[203] * 25 - 5) if len(feats) > 203 and is_attached else None
    host_toughness = round(feats[204] * 25 - 5) if len(feats) > 204 and is_attached else None
    host_is_creature = feats[205] > 0.5 if len(feats) > 205 and is_attached else False
    host_is_mine = feats[206] > 0.5 if len(feats) > 206 and is_attached else False
    host_cmc = round(feats[207] * 16) if len(feats) > 207 and is_attached else None

    # === COMBAT MATH [208-232] ===
    combat = None
    if "Creature" in types and len(feats) > 232:
        has_any = any(feats[i] != 0 for i in range(208, 233))
        if has_any:
            combat = {
                "can_attack": feats[208] > 0.5,
                "is_evasive": feats[209] > 0.5,
                "frac_can_block": round(feats[210], 2),
                "frac_this_kills": round(feats[211], 2),
                "frac_kills_this": round(feats[212], 2),
                "lethal_remaining": round(feats[213], 2),
                "power_vs_avg_t": round(feats[214], 2),
                "toughness_vs_avg_p": round(feats[215], 2),
                "first_strike_adv": feats[216] > 0.5,
                "has_deathtouch": feats[217] > 0.5,
                "has_indestructible": feats[218] > 0.5,
                "has_lifelink": feats[219] > 0.5,
                "has_trample": feats[220] > 0.5,
                "can_trade_up": feats[221] > 0.5,
                "can_trade_even": feats[222] > 0.5,
                "is_biggest": feats[223] > 0.5,
                "power_rank": round(feats[224], 2),
                "toughness_rank": round(feats[225], 2),
                "safe_attacker": feats[226] > 0.5,
                "must_double_block": feats[227] > 0.5,
                "best_blocker_power": round(feats[228] * 20),
                "best_blocker_toughness": round(feats[229] * 20),
                "n_profitable_blocks": round(feats[230] * 10, 1),
                "power_surplus": round(feats[231], 2),
                "needs_gang_block": feats[232] > 0.5,
            }

    return {
        "label": label, "types": types, "colors": colors,
        "cmc": cmc, "power": power, "toughness": toughness,
        "loyalty": loyalty,
        "tapped": tapped, "sick": sick,
        "attacking": attacking, "blocking": blocking,
        "face_down": face_down,
        "p1p1": p1p1, "m1m1": m1m1, "damage": damage,
        "loyalty_counters": loyalty_counters,
        "charge_counters": charge_counters,
        "other_counters": other_counters,
        "attachments": attachments,
        "keywords": kws + kws2, "zone": zone,
        # Ability info
        "primary_api": primary_api,
        "secondary_api": secondary_api,
        "has_activated": has_activated,
        "has_triggered": has_triggered,
        "has_mana_ability": has_mana_ability,
        "n_abilities": n_abilities,
        "est_damage": est_damage, "est_draw": est_draw,
        "est_life": est_life, "est_tokens": est_tokens,
        "requires_target": requires_target,
        "targets_creatures": targets_creatures,
        # Mana production
        "produces_mana": produces_mana,
        # Speed
        "is_instant_speed": is_instant_speed,
        "has_flash": has_flash,
        "is_modal": is_modal, "has_kicker": has_kicker,
        # Triggers
        "has_etb": has_etb, "has_death": has_death,
        "has_combat_trigger": has_combat_trigger,
        "has_upkeep": has_upkeep,
        # Trigger details
        "etb_api": etb_api, "etb_magnitude": etb_magnitude,
        "death_api": death_api, "death_magnitude": death_magnitude,
        "combat_trig_api": combat_trig_api,
        "combat_trig_magnitude": combat_trig_magnitude,
        # Mana cost
        "mana_cost_str": mana_cost_str,
        # Ownership
        "is_mine": is_mine, "is_opponents": is_opponents,
        # Pump
        "pump_power": pump_power,
        "pump_toughness": pump_toughness,
        # Aura/equipment host
        "is_attached": is_attached,
        "host_power": host_power,
        "host_toughness": host_toughness,
        "host_is_creature": host_is_creature,
        "host_is_mine": host_is_mine,
        "host_cmc": host_cmc,
        # Combat math
        "combat": combat,
    }


# ActionEncoder 64-dim feature layout
API_TYPES = [
    "DealDamage", "Draw", "Counter", "ChangeZone",
    "Pump", "PumpAll", "Destroy", "DestroyAll",
    "Sacrifice", "Discard", "GainLife", "LoseLife",
    "Token", "Animate", "Attach", "Tap",
    "Untap", "Mill", "Regenerate", "Protection",
    "Fight", "Charm", "Scry", "Explore",
    "AddOrRemoveCounter", "ManaReflected", "Mana",
    "ChangeTargets", "Fog", "ChangeZone",
]


# Human-readable API type descriptions
API_DESCRIPTIONS = {
    "DealDamage": "Deal damage",
    "Draw": "Draw cards",
    "Counter": "Counter spell",
    "ChangeZone": "Move card (bounce/exile/reanimate)",
    "ChangeZone2": "Move card",
    "Pump": "Pump creature (+X/+X)",
    "PumpAll": "Pump all creatures",
    "Destroy": "Destroy permanent",
    "DestroyAll": "Destroy all (board wipe)",
    "Sacrifice": "Force sacrifice",
    "Discard": "Force discard",
    "GainLife": "Gain life",
    "LoseLife": "Lose life",
    "Token": "Create token",
    "Animate": "Animate permanent",
    "Attach": "Attach (equip/enchant)",
    "Tap": "Tap permanent",
    "Untap": "Untap permanent",
    "Mill": "Mill cards",
    "Regenerate": "Regenerate",
    "Protection": "Grant protection",
    "Fight": "Fight creature",
    "Charm": "Charm (modal)",
    "Scry": "Scry",
    "Explore": "Explore",
    "AddOrRemoveCounter": "Add/remove counter",
    "ManaReflected": "Produce mana (reflected)",
    "Mana": "Produce mana",
    "ChangeTargets": "Change targets",
    "Fog": "Prevent combat damage",
}


def decode_action(feats):
    """Decode a 64-dim ActionEncoder feature vector."""
    if len(feats) < 18:
        return None

    # Check pass action (feature[63] = 1.0)
    if len(feats) > 63 and feats[63] > 0.5:
        return {"label": "PASS", "is_pass": True,
                "types": [], "colors": [], "cmc": 0,
                "sa_type": "pass", "api": None,
                "targets": False, "detail": "",
                "damage": 0, "cards_drawn": 0,
                "power": None, "toughness": None,
                "target_info": ""}

    type_names = ["Creature", "Instant", "Sorcery",
                  "Enchantment", "Artifact", "Planeswalker",
                  "Land"]
    types = [t for i, t in enumerate(type_names)
             if i < len(feats) and feats[i] > 0.5]

    color_chars = ["W", "U", "B", "R", "G", "C"]
    colors = [c for i, c in enumerate(color_chars)
              if 7+i < len(feats) and feats[7+i] > 0.5]

    cmc = int(round(feats[13] * 16)) if len(feats) > 13 \
        else 0

    sa_type = "spell" if feats[14] > 0.5 \
        else "activated" if feats[15] > 0.5 \
        else "triggered" if feats[16] > 0.5 \
        else "mana" if feats[17] > 0.5 \
        else "other"

    # API type [18-47]
    api = None
    for i, name in enumerate(API_TYPES):
        if 18+i < len(feats) and feats[18+i] > 0.5:
            api = name
            break

    # Targeting info [48-51]
    requires_target = feats[48] > 0.5 \
        if len(feats) > 48 else False
    n_targets = int(round(feats[49] * 5)) \
        if len(feats) > 49 else 0
    targets_creatures = feats[50] > 0.5 \
        if len(feats) > 50 else False
    targets_players = feats[51] > 0.5 \
        if len(feats) > 51 else False

    # Source P/T [52-53]
    power = None
    toughness = None
    if "Creature" in types and len(feats) > 53:
        power = int(round(feats[52] * 25 - 5))
        toughness = int(round(feats[53] * 25 - 5))

    # Estimated damage [54] and cards drawn [55]
    damage = 0
    cards_drawn = 0
    if len(feats) > 54:
        damage = int(round(feats[54] * 20))
    if len(feats) > 55:
        cards_drawn = int(round(feats[55] * 10))

    # Target polarity [56-59]
    can_target_own_creature = feats[56] > 0.5 if len(feats) > 56 else False
    can_target_opp_creature = feats[57] > 0.5 if len(feats) > 57 else False
    can_target_own_player = feats[58] > 0.5 if len(feats) > 58 else False
    can_target_opp_player = feats[59] > 0.5 if len(feats) > 59 else False

    # Build target description with polarity
    target_parts = []
    if requires_target:
        if can_target_own_creature and can_target_opp_creature:
            target_parts.append("→ any creature")
        elif can_target_opp_creature:
            target_parts.append("→ opp creature")
        elif can_target_own_creature:
            target_parts.append("→ own creature")
        elif targets_creatures:
            target_parts.append("→ creature")

        if targets_players:
            target_parts.append("/ player")
        if not target_parts:
            target_parts.append("→ target")
    target_info = " ".join(target_parts)

    # Build detail string
    detail_parts = []
    if api:
        desc = API_DESCRIPTIONS.get(api, api)
        detail_parts.append(desc)
    if damage > 0:
        detail_parts.append(f"{damage} dmg")
    if cards_drawn > 0:
        detail_parts.append(f"draw {cards_drawn}")
    if power is not None:
        detail_parts.append(f"{power}/{toughness}")
    if target_info:
        detail_parts.append(target_info)
    detail = " | ".join(detail_parts)

    # Build label
    color_str = "".join(colors) if colors else ""
    type_str = "/".join(types) if types else "?"
    label = f"{color_str} {type_str}" if color_str \
        else type_str
    label += f" [{cmc}]"
    if detail:
        label += f" — {detail}"

    return {
        "label": label, "is_pass": False,
        "types": types, "colors": colors, "cmc": cmc,
        "sa_type": sa_type, "api": api,
        "targets": requires_target,
        "target_info": target_info,
        "damage": damage, "cards_drawn": cards_drawn,
        "power": power, "toughness": toughness,
        "detail": detail,
        # Target polarity
        "can_target_own_creature": can_target_own_creature,
        "can_target_opp_creature": can_target_opp_creature,
        "can_target_own_player": can_target_own_player,
        "can_target_opp_player": can_target_opp_player,
    }


# ── Combat math tooltip ───────────────────────────

def format_combat_tooltip(info):
    """Format combat math data for tooltip display."""
    combat = info.get("combat")
    if not combat:
        return None

    lines = []
    label = info.get("label", "?")
    if info.get("power") is not None:
        label += f" {info['power']}/{info['toughness']}"
    lines.append(f"=== Combat Math: {label} ===")
    lines.append("")

    # Attack readiness
    flags = []
    if combat["can_attack"]:
        flags.append("CAN ATTACK")
    if combat["is_evasive"]:
        flags.append("EVASIVE")
    if combat["safe_attacker"]:
        flags.append("SAFE")
    if combat["is_biggest"]:
        flags.append("BIGGEST")
    if combat["must_double_block"]:
        flags.append("OVERWHELMS BLOCKERS")
    if combat["needs_gang_block"]:
        flags.append("NEEDS GANG BLOCK TO KILL")
    if flags:
        lines.append(" ".join(flags))
        lines.append("")

    # Combat keywords
    ckws = []
    if combat["has_deathtouch"]:
        ckws.append("Deathtouch")
    if combat["has_indestructible"]:
        ckws.append("Indestructible")
    if combat["has_lifelink"]:
        ckws.append("Lifelink")
    if combat["has_trample"]:
        ckws.append("Trample")
    if combat["first_strike_adv"]:
        ckws.append("First Strike Advantage")
    if ckws:
        lines.append("Combat: " + ", ".join(ckws))

    # Matchup stats
    lines.append(f"Kills {combat['frac_this_kills']:.0%} of opp creatures")
    lines.append(f"Killed by {combat['frac_kills_this']:.0%} of opp")
    if combat["frac_can_block"] > 0:
        lines.append(f"Blockable by {combat['frac_can_block']:.0%} of opp")

    # Trade info
    trades = []
    if combat["can_trade_up"]:
        trades.append("trades UP")
    if combat["can_trade_even"]:
        trades.append("trades even")
    if trades:
        lines.append("Can " + ", ".join(trades))

    # Relative strength
    lines.append("")
    lines.append(f"Power rank: {combat['power_rank']:.0%}")
    lines.append(f"Toughness rank: {combat['toughness_rank']:.0%}")
    lines.append(f"P vs avg T: {combat['power_vs_avg_t']:.2f}")
    lines.append(f"T vs avg P: {combat['toughness_vs_avg_p']:.2f}")

    # Blocker info
    if combat["best_blocker_power"] > 0:
        lines.append(f"Best blocker: {combat['best_blocker_power']:.0f}/{combat['best_blocker_toughness']:.0f}")

    # Defense
    if combat["n_profitable_blocks"] > 0:
        lines.append(f"Profitable blocks: {combat['n_profitable_blocks']:.0f}")

    lines.append(f"HP remaining: {combat['lethal_remaining']:.0%}")

    return "\n".join(lines)


def decode_global_combat(gf):
    """Decode global combat features [76-95] from global feature array."""
    if len(gf) <= 95:
        return None
    # Check if any combat globals are populated
    if not any(gf[i] != 0 for i in range(76, 96)):
        return None
    return {
        "my_attackable_power": round(gf[76] * 60),
        "my_evasive_power": round(gf[77] * 40),
        "opp_attackable_power": round(gf[78] * 60),
        "opp_evasive_power": round(gf[79] * 40),
        "lethal_on_board": gf[80] > 0.5,
        "opp_lethal_on_board": gf[81] > 0.5,
        "evasive_lethal": gf[82] > 0.5,
        "opp_evasive_lethal": gf[83] > 0.5,
        "my_safe_attack_power": round(gf[84] * 40),
        "board_power_adv": round((gf[85] * 120 - 60)),
        "board_toughness_adv": round((gf[86] * 120 - 60)),
        "creature_count_adv": round((gf[87] * 40 - 20)),
        "my_first_strikers": round(gf[88] * 10),
        "opp_first_strikers": round(gf[89] * 10),
        "my_deathtouchers": round(gf[90] * 10),
        "opp_deathtouchers": round(gf[91] * 10),
        "alpha_strike_kills": round(gf[92] * 20),
        "turns_to_lethal": round(gf[93] * 10),
        "opp_turns_to_lethal": round(gf[94] * 10),
        "combat_dominance": round(gf[95], 2),
    }


def format_global_combat(gc):
    """Format global combat features for display in the right panel."""
    if not gc:
        return ""
    lines = []
    lines.append("=== Combat Overview ===")
    lines.append("")

    # Lethal checks (most important)
    if gc["lethal_on_board"]:
        lines.append("!! LETHAL ON BOARD !!")
    if gc["evasive_lethal"]:
        lines.append("!! EVASIVE LETHAL !!")
    if gc["opp_lethal_on_board"]:
        lines.append("!! OPP HAS LETHAL !!")
    if gc["opp_evasive_lethal"]:
        lines.append("!! OPP EVASIVE LETHAL !!")

    # Power summary
    lines.append(f"My attack power: {gc['my_attackable_power']}"
                 f" (safe: {gc['my_safe_attack_power']},"
                 f" evasive: {gc['my_evasive_power']})")
    lines.append(f"Opp attack power: {gc['opp_attackable_power']}"
                 f" (evasive: {gc['opp_evasive_power']})")
    lines.append("")

    # Board advantage
    pwr = gc["board_power_adv"]
    tgh = gc["board_toughness_adv"]
    cnt = gc["creature_count_adv"]
    lines.append(f"Board advantage:")
    lines.append(f"  Power: {pwr:+d}  Toughness: {tgh:+d}"
                 f"  Count: {cnt:+d}")

    # Special creatures
    specials = []
    if gc["my_first_strikers"]:
        specials.append(f"My FS: {gc['my_first_strikers']}")
    if gc["opp_first_strikers"]:
        specials.append(f"Opp FS: {gc['opp_first_strikers']}")
    if gc["my_deathtouchers"]:
        specials.append(f"My DT: {gc['my_deathtouchers']}")
    if gc["opp_deathtouchers"]:
        specials.append(f"Opp DT: {gc['opp_deathtouchers']}")
    if specials:
        lines.append("  " + "  ".join(specials))

    lines.append("")
    lines.append(f"Alpha strike kills: {gc['alpha_strike_kills']}")
    lines.append(f"Turns to lethal: {gc['turns_to_lethal']}")
    lines.append(f"Opp turns to lethal: {gc['opp_turns_to_lethal']}")
    lines.append(f"Combat dominance: {gc['combat_dominance']:.2f}")

    return "\n".join(lines)


class CardTooltip:
    """Tooltip that shows combat math on card mouseover."""

    def __init__(self, widget, text_func):
        self.widget = widget
        self.text_func = text_func
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        text = self.text_func()
        if not text:
            return
        self.hide()
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=text, justify=tk.LEFT,
                       bg="#1e1e2e", fg="#cdd6f4",
                       font=("Consolas", 9),
                       relief=tk.SOLID, borderwidth=1,
                       padx=8, pady=6)
        lbl.pack()

    def hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


# ── Card rendering (MTG card style) ───────────────

CARD_W, CARD_H = 100, 140
ACTION_W, ACTION_H = 100, 140


def _load_fonts():
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 8)
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 10)
    except (IOError, OSError):
        font_sm = ImageFont.load_default()
        font_md = font_sm
        font_lg = font_sm
    return font_sm, font_md, font_lg


def draw_card_image(info, highlight=None):
    """Draw a card as PIL image in MTG card style."""
    if not HAS_PIL:
        return None

    c = info["colors"][0] if info["colors"] else "C"
    bg_color, fg_color = COLOR_HEX.get(c, COLOR_HEX["C"])

    # Border
    if highlight == "attack":
        border = "#00ff00"
    elif highlight == "block":
        border = "#4488ff"
    else:
        border = "#333333"

    img = Image.new("RGB", (CARD_W, CARD_H), border)
    draw = ImageDraw.Draw(img)

    # Card body
    m = 3  # border margin
    draw.rectangle([m, m, CARD_W-m-1, CARD_H-m-1], fill=bg_color)

    font_sm, font_md, font_lg = _load_fonts()

    y = 4

    # Title bar: type + mana cost
    type_str = "/".join(info["types"])[:12]
    draw.text((4, y), type_str, fill=fg_color, font=font_md)
    mana_str = info.get("mana_cost_str", "")
    if mana_str:
        draw.text((CARD_W - 4, y), mana_str,
                  fill=fg_color, font=font_md, anchor='ra')
    else:
        cmc_str = f"{{{info['cmc']}}}"
        draw.text((CARD_W - 4, y), cmc_str,
                  fill=fg_color, font=font_md, anchor='ra')
    y += 12

    # Color identity
    color_str = "".join(info["colors"]) if info["colors"] else "Colorless"
    draw.text((4, y), color_str, fill=fg_color, font=font_sm)
    y += 10

    # Separator
    draw.line([(4, y), (CARD_W-4, y)], fill=fg_color, width=1)
    y += 3

    # Art area — show ability info
    art_h = 30
    darker = _darken(bg_color, 0.8)
    draw.rectangle([4, y, CARD_W-4, y+art_h], fill=darker)

    if "Land" in info["types"]:
        draw.text((6, y+2), info["label"],
                  fill=fg_color, font=font_lg)
        if info.get("produces_mana"):
            draw.text((6, y+14),
                      f"Tap: {' '.join(info['produces_mana'])}",
                      fill="#ffdd00", font=font_sm)
    elif info.get("primary_api"):
        api_desc = API_DESCRIPTIONS.get(
            info["primary_api"], info["primary_api"])
        draw.text((6, y+2), api_desc[:16],
                  fill=fg_color, font=font_sm)
        # Effect magnitude
        effects = []
        if info.get("est_damage"):
            effects.append(f"{info['est_damage']}dmg")
        if info.get("est_draw"):
            effects.append(f"draw {info['est_draw']}")
        if info.get("est_life"):
            effects.append(f"+{info['est_life']}life")
        if info.get("est_tokens"):
            effects.append(f"{info['est_tokens']}tok")
        if effects:
            draw.text((6, y+12), " ".join(effects),
                      fill="#ffdd00", font=font_sm)
        # Secondary ability
        if info.get("secondary_api"):
            api2 = API_DESCRIPTIONS.get(
                info["secondary_api"], info["secondary_api"])
            draw.text((6, y+22), f"+ {api2[:14]}",
                      fill="#aaaaaa", font=font_sm)
    y += art_h + 3

    # Separator
    draw.line([(4, y), (CARD_W-4, y)], fill=fg_color, width=1)
    y += 3

    # Keywords (compact, up to 4)
    all_kws = info.get("keywords", [])
    for kw in all_kws[:4]:
        draw.text((4, y), kw, fill=fg_color, font=font_sm)
        y += 9
    if len(all_kws) > 4:
        draw.text((4, y), f"+{len(all_kws)-4} more",
                  fill="#888888", font=font_sm)
        y += 9

    # Triggers line
    triggers = []
    if info.get("has_etb"):
        triggers.append("ETB")
    if info.get("has_death"):
        triggers.append("Dies")
    if info.get("has_combat_trigger"):
        triggers.append("Combat")
    if info.get("has_upkeep"):
        triggers.append("Upkeep")
    if triggers:
        draw.text((4, y), " ".join(triggers),
                  fill="#cba6f7", font=font_sm)
        y += 9

    # Speed flags
    speed = []
    if info.get("has_flash"):
        speed.append("Flash")
    elif info.get("is_instant_speed"):
        speed.append("Instant")
    if info.get("is_modal"):
        speed.append("Modal")
    if info.get("has_kicker"):
        speed.append("Kicker")
    if speed:
        draw.text((4, y), " ".join(speed),
                  fill="#89b4fa", font=font_sm)
        y += 9

    # Mana production (for non-lands)
    if info.get("produces_mana") and "Land" not in info["types"]:
        draw.text((4, y),
                  f"Tap: {' '.join(info['produces_mana'])}",
                  fill="#ffdd00", font=font_sm)
        y += 9

    # Counters
    p1p1 = info.get("p1p1", 0)
    m1m1 = info.get("m1m1", 0)
    if p1p1 > 0:
        draw.text((4, y), f"+{p1p1}/+{p1p1}",
                  fill="#ffdd00", font=font_sm)
        y += 9
    if m1m1 > 0:
        draw.text((4, y), f"-{m1m1}/-{m1m1}",
                  fill="#ff4444", font=font_sm)
        y += 9
    if info.get("charge_counters", 0) > 0:
        draw.text((4, y),
                  f"Charge: {info['charge_counters']}",
                  fill="#ffdd00", font=font_sm)
        y += 9

    # Attachments
    if info.get("attachments", 0) > 0:
        draw.text((4, y),
                  f"{info['attachments']} attached",
                  fill="#aaaaaa", font=font_sm)
        y += 9

    # Damage marked
    dmg = info.get("damage", 0)
    if dmg > 0:
        draw.text((4, y), f"{dmg} dmg",
                  fill="#ff6666", font=font_sm)
        y += 9

    # Loyalty (planeswalkers)
    loyalty = info.get("loyalty")
    if loyalty is not None and loyalty > 0:
        draw.text((4, y), f"Loyalty: {loyalty}",
                  fill="#ccccff", font=font_sm)
        y += 9

    # State flags
    if info.get("tapped"):
        draw.text((4, y), "TAPPED",
                  fill="#ff6666", font=font_sm)
        y += 9
    if info.get("sick") and info.get("zone") == "Battlefield":
        draw.text((4, y), "SICK",
                  fill="#ffaa44", font=font_sm)
        y += 9
    if info.get("attacking"):
        draw.text((4, y), "ATTACKING",
                  fill="#00ff00", font=font_sm)
        y += 9
    if info.get("blocking"):
        draw.text((4, y), "BLOCKING",
                  fill="#4488ff", font=font_sm)
        y += 9

    # P/T box (bottom right for creatures)
    if info.get("power") is not None:
        pt = f"{info['power']}/{info['toughness']}"
        box_w = 30
        bx = CARD_W - box_w - 4
        by = CARD_H - 18
        draw.rectangle([bx, by, CARD_W-4, CARD_H-4],
                       fill="#000000", outline=fg_color)
        draw.text((bx+3, by+1), pt,
                  fill="#ffffff", font=font_lg)

    # CMC box (bottom left)
    cmc = info.get("cmc", 0)
    if cmc > 0 and "Land" not in info.get("types", []):
        draw.rectangle([4, CARD_H-18, 24, CARD_H-4],
                       fill="#444444", outline=fg_color)
        draw.text((7, CARD_H-17), str(cmc),
                  fill="#ffffff", font=font_lg)

    return img


def _darken(hex_color, factor):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def draw_action_card_image(info, is_chosen=False,
                           is_pass=False):
    """Draw a priority action candidate as a card."""
    if not HAS_PIL:
        return None

    if is_pass or info.get("is_pass"):
        # Pass action — distinct style
        border = "#f9e2af" if is_chosen else "#585b70"
        img = Image.new("RGB", (ACTION_W, ACTION_H),
                        border)
        d = ImageDraw.Draw(img)
        m = 3
        d.rectangle([m, m, ACTION_W-m-1, ACTION_H-m-1],
                     fill="#313244")
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSansMono-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        d.text((ACTION_W//2, ACTION_H//2 - 8), "PASS",
               fill="#f9e2af" if is_chosen else "#6c7086",
               font=font, anchor='mt')
        if is_chosen:
            d.text((ACTION_W//2, ACTION_H//2 + 12),
                   "CHOSEN",
                   fill="#a6e3a1", font=font, anchor='mt')
        return img

    c = info["colors"][0] if info.get("colors") else "C"
    bg_color, fg_color = COLOR_HEX.get(c, COLOR_HEX["C"])

    if is_chosen:
        border = "#a6e3a1"
    else:
        border = "#333333"

    img = Image.new("RGB", (ACTION_W, ACTION_H), border)
    d = ImageDraw.Draw(img)
    m = 3
    d.rectangle([m, m, ACTION_W-m-1, ACTION_H-m-1],
                 fill=bg_color)

    font_sm, font_md, font_lg = _load_fonts()

    y = 4

    # Type + CMC header
    type_str = "/".join(
        info.get("types", []))[:12] or "?"
    d.text((4, y), type_str, fill=fg_color, font=font_md)
    cmc = info.get("cmc", 0)
    d.text((ACTION_W - 22, y), f"{{{cmc}}}",
           fill=fg_color, font=font_md)
    y += 12

    # Color + spell/ability type
    color_str = "".join(
        info.get("colors", [])) or "Colorless"
    sa = info.get("sa_type", "")
    d.text((4, y), f"{color_str} {sa}",
           fill=fg_color, font=font_sm)
    y += 10

    # Separator
    d.line([(4, y), (ACTION_W-4, y)],
           fill=fg_color, width=1)
    y += 3

    # Effect area — API type description
    darker = _darken(bg_color, 0.8)
    d.rectangle([4, y, ACTION_W-4, y+42], fill=darker)

    api = info.get("api")
    if api:
        desc = API_DESCRIPTIONS.get(api, api)
        # Wrap long descriptions
        if len(desc) > 14:
            parts = desc.split(" ", 1)
            d.text((6, y+3), parts[0],
                   fill=fg_color, font=font_md)
            if len(parts) > 1:
                d.text((6, y+13), parts[1][:16],
                       fill=fg_color, font=font_sm)
        else:
            d.text((6, y+3), desc,
                   fill=fg_color, font=font_md)

    # Damage / draw info
    dmg = info.get("damage", 0)
    cards = info.get("cards_drawn", 0)
    ey = y + 24
    if dmg > 0:
        d.text((6, ey), f"{dmg} damage",
               fill="#ff6666", font=font_md)
        ey += 10
    if cards > 0:
        d.text((6, ey), f"Draw {cards}",
               fill="#89b4fa", font=font_md)
        ey += 10

    y += 45

    # Separator
    d.line([(4, y), (ACTION_W-4, y)],
           fill=fg_color, width=1)
    y += 3

    # Target info
    ti = info.get("target_info", "")
    if ti:
        d.text((4, y), ti, fill=fg_color, font=font_sm)
        y += 10

    # P/T for creatures
    pw = info.get("power")
    if pw is not None:
        pt = f"{pw}/{info.get('toughness', 0)}"
        box_w = 30
        bx = ACTION_W - box_w - 4
        by = ACTION_H - 18
        d.rectangle([bx, by, ACTION_W-4, ACTION_H-4],
                     fill="#000000", outline=fg_color)
        d.text((bx+3, by+1), pt,
               fill="#ffffff", font=font_lg)

    # "CHOSEN" marker
    if is_chosen:
        d.text((4, ACTION_H - 16), ">> CHOSEN",
               fill="#a6e3a1", font=font_md)

    return img


# ── Data loading ───────────────────────────────────

def load_samples(data_dir, max_samples=500,
                  rl_only=False, source_tag="heuristic"):
    path = Path(data_dir)
    files = sorted(path.glob("traj_*.jsonl"))
    if rl_only:
        # Only load RL player trajectories
        files = [f for f in files
                 if '_RL_' in f.name]
    random.shuffle(files)

    samples = []
    for f in files:
        if len(samples) >= max_samples:
            break
        try:
            lines = open(f).readlines()
            if len(lines) < 2:
                continue
            header = json.loads(lines[0])
            won = header.get("won", False)

            for line in lines[1:]:
                rec = json.loads(line)
                dt = rec.get("decisionType", "")
                if dt not in ("DECLARE_ATTACKERS",
                              "DECLARE_BLOCKERS",
                              "PRIORITY_ACTION",
                              "TARGET_SELECTION",
                              "CARD_SELECTION",
                              "MULLIGAN",
                              "BINARY_CHOICE"):
                    continue
                cand = rec.get("candidateFeatures", [])
                # Binary and mulligan may have no/few candidates
                if dt not in ("BINARY_CHOICE", "MULLIGAN") \
                        and len(cand) < 1:
                    continue

                samples.append({
                    "type": dt,
                    "info": rec.get("contextInfo", ""),
                    "global_features": np.array(
                        rec.get("globalFeatures", []),
                        dtype=np.float32),
                    "game_state_flat": np.array(
                        rec.get("gameStateFlat", []),
                        dtype=np.float32),
                    "candidates": cand,
                    "selected": rec.get(
                        "selectedIndices", []),
                    "won": won,
                    "source": source_tag,
                })
                if len(samples) >= max_samples:
                    break
        except Exception:
            pass
    return samples


# ── Main Viewer ────────────────────────────────────

class GameStateViewer:
    # Viewing modes
    MODE_RL_REPLAY = "rl_replay"        # PPO trajectories — RL decisions at collection time
    MODE_HEURISTIC = "heuristic"        # Base trajectories — heuristic decisions

    def __init__(self, root, samples, model, device,
                 model_path=None, data_dir=None):
        self.root = root
        self.all_samples = samples
        self.samples = samples
        self.model = model
        self.device = device
        self.model_path = model_path
        self.data_dir = data_dir
        self.idx = 0
        self.card_photos = []  # keep references
        self._pending_show = None  # for buffered updates

        # Determine initial mode from data
        has_rl = any(s.get("source") == "ppo" for s in samples)
        has_heur = any(s.get("source") == "heuristic" for s in samples)
        self.mode = self.MODE_RL_REPLAY if has_rl else self.MODE_HEURISTIC

        root.title("MTG RL — Game State Visualizer")
        root.geometry("1500x950")
        root.configure(bg="#1e1e2e")

        self._build(root)
        self._apply_mode_filter()
        if self.samples:
            self._show()

    def _build(self, root):
        # Nav bar
        nav = tk.Frame(root, bg="#1e1e2e")
        nav.pack(fill=tk.X, padx=10, pady=5)

        # Mode toggle — prominent indicator
        self.mode_v = tk.StringVar()
        self.mode_lbl = tk.Label(nav, textvariable=self.mode_v,
                                  bg="#1e1e2e",
                                  font=("Consolas", 11, "bold"),
                                  padx=8, pady=2)
        self.mode_lbl.pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(nav, text="Switch Mode",
                   command=self._toggle_mode).pack(side=tk.LEFT, padx=3)

        tk.Frame(nav, bg="#45475a", width=2).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Random", command=self._rand).pack(side=tk.LEFT, padx=3)

        tk.Frame(nav, bg="#45475a", width=2).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(nav, text="Attacks", command=self._filter_attacks).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Blocks", command=self._filter_blocks).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Priority", command=self._filter_priority).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Pri Pass", command=self._filter_priority_pass).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Pri Play", command=self._filter_priority_play).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Target", command=self._filter_target).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="CardSel", command=self._filter_card_select).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Mulligan", command=self._filter_mulligan).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav, text="All", command=self._filter_all).pack(side=tk.LEFT, padx=3)

        tk.Frame(nav, bg="#45475a", width=2).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(nav, text="Disagree",
                   command=self._filter_disagree).pack(
                       side=tk.LEFT, padx=3)

        tk.Frame(nav, bg="#45475a", width=2).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(nav, text="Reload Model",
                   command=self._reload_model).pack(
                       side=tk.LEFT, padx=3)
        ttk.Button(nav, text="Reload Data",
                   command=self._reload_data).pack(
                       side=tk.LEFT, padx=3)

        self.model_v = tk.StringVar(
            value=os.path.basename(self.model_path)
            if self.model_path else "no model")
        tk.Label(nav, textvariable=self.model_v,
                 bg="#1e1e2e", fg="#a6e3a1",
                 font=("Consolas", 9)).pack(
                     side=tk.RIGHT, padx=5)

        self.nav_v = tk.StringVar(value="0/0")
        tk.Label(nav, textvariable=self.nav_v, bg="#1e1e2e",
                 fg="#a6adc8", font=("Consolas", 11)).pack(side=tk.LEFT, padx=15)

        self.type_v = tk.StringVar()
        tk.Label(nav, textvariable=self.type_v, bg="#1e1e2e",
                 fg="#f9e2af", font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=5)

        self.info_v = tk.StringVar()
        tk.Label(nav, textvariable=self.info_v, bg="#1e1e2e",
                 fg="#89b4fa", font=("Consolas", 11)).pack(side=tk.LEFT, padx=10)

        self.outcome_v = tk.StringVar()
        self.outcome_lbl = tk.Label(nav, textvariable=self.outcome_v,
                                     bg="#1e1e2e", font=("Consolas", 14, "bold"))
        self.outcome_lbl.pack(side=tk.RIGHT, padx=10)

        # Status bar
        self.status_v = tk.StringVar()
        tk.Label(root, textvariable=self.status_v, bg="#181825",
                 fg="#cdd6f4", font=("Consolas", 11), anchor="w",
                 padx=10, pady=4).pack(fill=tk.X)

        # Main: eval bar | board | predictions
        main = tk.Frame(root, bg="#1e1e2e")
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Eval bar (lichess-style vertical bar)
        eval_frame = tk.Frame(main, bg="#1e1e2e", width=36)
        eval_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        eval_frame.pack_propagate(False)

        self.eval_label = tk.Label(eval_frame, text="",
                                    bg="#1e1e2e", fg="#cdd6f4",
                                    font=("Consolas", 9))
        self.eval_label.pack(side=tk.TOP, pady=(2, 2))

        self.eval_canvas = tk.Canvas(eval_frame, width=28,
                                      bg="#333333", highlightthickness=0)
        self.eval_canvas.pack(fill=tk.Y, expand=True, padx=4)
        self.eval_canvas.bind("<Configure>",
                              lambda e: self._update_eval_bar(self._last_value))
        self._last_value = None

        self.eval_label_btm = tk.Label(eval_frame, text="",
                                        bg="#1e1e2e", fg="#cdd6f4",
                                        font=("Consolas", 9))
        self.eval_label_btm.pack(side=tk.BOTTOM, pady=(2, 2))

        # Board areas
        board_col = tk.Frame(main, bg="#1e1e2e")
        board_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Opponent board
        self._make_section(board_col, "Opponent's Board", "#f38ba8")
        self.opp_creatures = tk.Frame(board_col, bg="#252535")
        self.opp_creatures.pack(fill=tk.X, padx=5)
        self.opp_lands = tk.Frame(board_col, bg="#202030")
        self.opp_lands.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(fill=tk.X, pady=3)

        # My board
        self._make_section(board_col, "My Board", "#a6e3a1")
        self.my_creatures = tk.Frame(board_col, bg="#252535")
        self.my_creatures.pack(fill=tk.X, padx=5)
        self.my_lands = tk.Frame(board_col, bg="#202030")
        self.my_lands.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(fill=tk.X, pady=3)

        # Stack
        self._make_section(board_col, "Stack", "#cba6f7")
        self.stack_frame = tk.Frame(board_col, bg="#252535")
        self.stack_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(
            fill=tk.X, pady=3)

        # My hand
        self._make_section(board_col, "My Hand", "#f9e2af")
        self.hand_frame = tk.Frame(board_col, bg="#252535")
        self.hand_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        tk.Frame(board_col, bg="#45475a", height=2).pack(
            fill=tk.X, pady=3)

        # Priority candidates (shown for PRIORITY_ACTION)
        self._make_section(board_col,
                           "Candidates", "#fab387")
        self.priority_frame = tk.Frame(
            board_col, bg="#252535")
        self.priority_frame.pack(
            fill=tk.X, padx=5, pady=(0, 5))

        # Right: model predictions
        pred_col = tk.Frame(main, bg="#181825", width=320)
        pred_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        pred_col.pack_propagate(False)

        self.pred_title_v = tk.StringVar(value="Model Prediction")
        tk.Label(pred_col, textvariable=self.pred_title_v,
                 bg="#181825", fg="#89b4fa",
                 font=("Consolas", 12, "bold")).pack(padx=5, pady=5)

        self.pred_text = tk.Text(pred_col, bg="#181825", fg="#cdd6f4",
                                  font=("Consolas", 10), wrap=tk.WORD,
                                  state=tk.DISABLED)
        self.pred_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _make_section(self, parent, title, color):
        tk.Label(parent, text=title, bg="#1e1e2e",
                 fg=color, font=("Consolas", 10, "bold"),
                 anchor="w").pack(fill=tk.X, padx=5)

    def _toggle_mode(self):
        """Switch between RL Replay and Heuristic Analysis modes."""
        if self.mode == self.MODE_RL_REPLAY:
            self.mode = self.MODE_HEURISTIC
        else:
            self.mode = self.MODE_RL_REPLAY
        self._apply_mode_filter()
        if self.samples:
            self._show()

    def _apply_mode_filter(self):
        """Filter samples to current mode and update mode indicator."""
        if self.mode == self.MODE_RL_REPLAY:
            filtered = [s for s in self.all_samples
                        if s.get("source") == "ppo"]
            self.mode_v.set("RL REPLAY")
            self.mode_lbl.configure(fg="#a6e3a1", bg="#2a3a2a")
        else:
            filtered = [s for s in self.all_samples
                        if s.get("source") == "heuristic"]
            self.mode_v.set("HEURISTIC ANALYSIS")
            self.mode_lbl.configure(fg="#f9e2af", bg="#3a3520")

        if filtered:
            self.samples = filtered
        else:
            # No data for this mode — keep all and warn
            self.samples = self.all_samples
            self.mode_v.set(self.mode_v.get() + " (no data — showing all)")
        self.idx = 0

    def _get_choice_label(self):
        """Label for the recorded choice based on mode."""
        if self.mode == self.MODE_RL_REPLAY:
            return "RL chose"
        else:
            return "Heuristic"

    def _get_compare_label(self):
        """Label for the model comparison based on mode."""
        if self.mode == self.MODE_RL_REPLAY:
            return "Current model"
        else:
            return "Model would"

    def _mode_samples(self):
        """Get samples for the current mode."""
        tag = "ppo" if self.mode == self.MODE_RL_REPLAY else "heuristic"
        filtered = [s for s in self.all_samples
                    if s.get("source") == tag]
        return filtered if filtered else self.all_samples

    def _filter_by_type(self, dtype):
        f = [s for s in self._mode_samples() if s["type"] == dtype]
        if f:
            self.samples = f
        else:
            # Try the other mode
            other = [s for s in self.all_samples if s["type"] == dtype]
            if other:
                self.samples = other
                # Auto-switch mode
                if other[0].get("source") == "ppo":
                    self.mode = self.MODE_RL_REPLAY
                else:
                    self.mode = self.MODE_HEURISTIC
                self._apply_mode_filter()
                self.samples = [s for s in self._mode_samples()
                                if s["type"] == dtype]
            else:
                self.samples = self._mode_samples()
        self.idx = 0
        self._show()

    def _filter_attacks(self):
        self._filter_by_type("DECLARE_ATTACKERS")

    def _filter_blocks(self):
        self._filter_by_type("DECLARE_BLOCKERS")

    def _filter_target(self):
        self._filter_by_type("TARGET_SELECTION")

    def _filter_card_select(self):
        self._filter_by_type("CARD_SELECTION")

    def _filter_mulligan(self):
        self._filter_by_type("MULLIGAN")

    def _filter_priority(self):
        f = [s for s in self._mode_samples()
             if s["type"] == "PRIORITY_ACTION"]
        self.samples = f if f else self._mode_samples()
        self.idx = 0
        self._show()

    def _filter_priority_pass(self):
        f = [s for s in self._mode_samples()
             if s["type"] == "PRIORITY_ACTION"
             and s["selected"]
             and s["selected"][0] == len(
                 s["candidates"]) - 1]
        self.samples = f if f else self._mode_samples()
        self.idx = 0
        self._show()

    def _filter_priority_play(self):
        f = [s for s in self._mode_samples()
             if s["type"] == "PRIORITY_ACTION"
             and s["selected"]
             and s["selected"][0] < len(
                 s["candidates"]) - 1]
        self.samples = f if f else self._mode_samples()
        self.idx = 0
        self._show()

    def _get_model_pick(self, s):
        """Get the model's chosen action for a sample.
        Returns model's pick to compare with heuristic."""
        if self.model is None:
            return None
        try:
            gf = s["global_features"]
            flat = s["game_state_flat"]
            g = np.zeros(GLOBAL_DIM, dtype=np.float32)
            gl = min(len(gf), GLOBAL_DIM)
            if gl > 0:
                g[:gl] = gf[:gl]
            np.clip(g, -10, 10, out=g)
            _, zones, masks = parse_game_state(flat, g)

            with torch.no_grad():
                t = lambda x: torch.from_numpy(
                    x).unsqueeze(0).to(self.device)
                state = self.model.encode_state(
                    t(g),
                    t(zones["my_board"]),
                    t(masks["my_board_mask"]),
                    t(zones["opp_board"]),
                    t(masks["opp_board_mask"]),
                    t(zones["hand"]),
                    t(masks["hand_mask"]),
                    t(zones["my_gy"]),
                    t(masks["my_gy_mask"]),
                    t(zones["opp_gy"]),
                    t(masks["opp_gy_mask"]),
                    t(zones["stack"]),
                    t(masks["stack_mask"]))

                dt = s.get("type", "")
                candidates = s.get("candidates", [])
                if not candidates:
                    return None

                if dt == "PRIORITY_ACTION":
                    n = len(candidates)
                    af_t = torch.zeros(
                        1, n, 64, device=self.device)
                    am_t = torch.zeros(
                        1, n, dtype=torch.bool,
                        device=self.device)
                    for j, cf in enumerate(candidates):
                        cl = min(len(cf), 64)
                        af_t[0, j, :cl] = torch.tensor(
                            cf[:cl], dtype=torch.float32)
                        am_t[0, j] = True
                    logits = self.model.priority_head(
                        state, af_t, am_t)
                    return logits[0].argmax().item()

                elif dt == "DECLARE_ATTACKERS":
                    n = len(candidates)
                    cf_t = torch.zeros(
                        1, n, CARD_DIM, device=self.device)
                    cm_t = torch.ones(
                        1, n, dtype=torch.bool,
                        device=self.device)
                    for j, cf in enumerate(candidates):
                        cl = min(len(cf), CARD_DIM)
                        cf_t[0, j, :cl] = torch.tensor(
                            cf[:cl], dtype=torch.float32)
                    logits = self.model.attack_head(
                        state, cf_t, cm_t)
                    probs = torch.sigmoid(logits)
                    return set(j for j in range(n)
                               if probs[0, j].item() > 0.5)

        except Exception:
            return None
        return None

    def _filter_disagree(self):
        """Show only samples where current model disagrees
        with the recorded choice."""
        if self.model is None:
            return
        disagree = []
        for s in self._mode_samples():
            dt = s.get("type", "")
            pick = self._get_model_pick(s)
            if pick is None:
                continue

            if dt == "PRIORITY_ACTION":
                sel = s["selected"]
                recorded_pick = sel[0] if sel else -1
                if pick != recorded_pick:
                    disagree.append(s)
            elif dt == "DECLARE_ATTACKERS":
                recorded_pick = set(s.get("selected", []))
                if pick != recorded_pick:
                    disagree.append(s)

        if disagree:
            self.samples = disagree
        else:
            self.samples = self._mode_samples()
        self.idx = 0
        self._show()

    def _reload_model(self):
        """Reload model from disk (picks up PPO updates)."""
        if not self.model_path:
            return
        # Try best available in order of preference
        ckpt_dir = os.path.dirname(self.model_path)
        candidates = [
            os.path.join(ckpt_dir, 'best_ppo_model.pt'),
            os.path.join(ckpt_dir, 'model_with_decisions.pt'),
            os.path.join(ckpt_dir, 'best_priority_model.pt'),
            os.path.join(ckpt_dir, 'best_attack_model.pt'),
            os.path.join(ckpt_dir, 'best_block_model.pt'),
            self.model_path,
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    self.model = MTGModel.load(
                        path, device=self.device)
                    self.model.eval()
                    self.model_path = path
                    self.model_v.set(
                        os.path.basename(path) +
                        " (reloaded)")
                    print(f"Reloaded model: {path}",
                          flush=True)
                    if self.samples:
                        self._show()
                    return
                except Exception as e:
                    print(f"Skipping {os.path.basename(path)}: {e}",
                          flush=True)

    def _reload_data(self):
        """Reload trajectory data from disk."""
        if not self.data_dir:
            return
        # Also check PPO trajectories
        dirs = [self.data_dir]
        ppo_dir = os.path.join(
            os.path.dirname(self.data_dir),
            'ppo_trajectories')
        if os.path.isdir(ppo_dir):
            dirs.append(ppo_dir)
        ppo_eval = ppo_dir + '_eval'
        if os.path.isdir(ppo_eval):
            dirs.append(ppo_eval)

        all_samples = []
        for d in dirs:
            is_ppo = 'ppo' in d
            tag = "ppo" if is_ppo else "heuristic"
            all_samples.extend(
                load_samples(d, max_samples=100,
                             rl_only=is_ppo,
                             source_tag=tag))
        if all_samples:
            self.all_samples = all_samples
            self._apply_mode_filter()
            self._show()

    def _filter_all(self):
        self.samples = self._mode_samples()
        self.idx = 0
        self._show()

    def _prev(self):
        self.idx = max(0, self.idx - 1)
        self._schedule_show()

    def _next(self):
        self.idx = min(len(self.samples) - 1, self.idx + 1)
        self._schedule_show()

    def _rand(self):
        self.idx = random.randint(0, len(self.samples) - 1)
        self._schedule_show()

    def _schedule_show(self):
        """Buffer rapid navigation — only render the last request."""
        if self._pending_show is not None:
            self.root.after_cancel(self._pending_show)
        self._pending_show = self.root.after(16, self._do_show)

    def _do_show(self):
        self._pending_show = None
        self._show()

    def _show(self):
        s = self.samples[self.idx]
        new_photos = []  # build new list, swap at end

        self.nav_v.set(f"{self.idx+1}/{len(self.samples)}")
        dt = s.get("type", "?")
        type_labels = {
            "DECLARE_ATTACKERS": "ATTACK",
            "DECLARE_BLOCKERS": "BLOCK",
            "PRIORITY_ACTION": "PRIORITY",
            "TARGET_SELECTION": "TARGET",
            "CARD_SELECTION": "CARD SELECT",
            "MULLIGAN": "MULLIGAN",
            "BINARY_CHOICE": "BINARY",
        }
        self.type_v.set(type_labels.get(dt, dt))
        self.info_v.set(s["info"])
        won = s["won"]
        self.outcome_v.set("WON" if won else "LOST")
        self.outcome_lbl.configure(fg="#a6e3a1" if won else "#f38ba8")

        gf = s["global_features"]
        flat = s["game_state_flat"]

        # Status
        PHASE_NAMES = [
            "Untap", "Upkeep", "Draw", "Main 1",
            "Begin Combat", "Declare Attackers",
            "Declare Blockers", "First Strike Dmg",
            "Combat Damage", "End Combat",
            "Main 2", "End of Turn", "Cleanup",
        ]
        if len(gf) >= 35:
            turn = int(round(gf[4] * 30))
            active = "My turn" if gf[5] > 0.5 \
                else "Opp turn"
            phase = "?"
            for i, pn in enumerate(PHASE_NAMES):
                if 6+i < len(gf) and gf[6+i] > 0.5:
                    phase = pn
                    break
            lands_untap = int(round(gf[29] * 15))
            stack_size = int(round(gf[32] * 10))
            self.status_v.set(
                f"Turn {turn} | {active} | {phase}  ||  "
                f"Life: {gf[0]*50-10:.0f} vs "
                f"{gf[1]*50-10:.0f}  |  "
                f"Hand: {gf[19]*15:.0f} vs "
                f"{gf[20]*15:.0f}  |  "
                f"Creatures: {gf[23]*20:.0f} vs "
                f"{gf[24]*20:.0f}  |  "
                f"Lands untapped: {lands_untap}  |  "
                f"Stack: {stack_size}")

        # Parse zones
        card_dim = CARD_DIM
        zones = {}
        offset = GLOBAL_DIM
        for zname, count in [("my_board", 40), ("opp_board", 40),
                              ("my_hand", 15), ("my_gy", 20),
                              ("opp_gy", 20), ("stack", 10)]:
            cards = []
            for j in range(count):
                start = offset + j * card_dim
                end = start + card_dim
                if end <= len(flat):
                    cf = flat[start:end]
                    if np.any(cf != 0):
                        info = decode_card(cf)
                        if info:
                            cards.append(info)
            zones[zname] = cards
            offset += count * card_dim

        # Split my board creatures — mark which are attacking
        selected = set(s.get("selected", []))
        my_board = zones.get("my_board", [])
        my_creatures = [c for c in my_board if "Land" not in c["types"]]
        my_lands = [c for c in my_board if "Land" in c["types"]]

        opp_board = zones.get("opp_board", [])
        opp_creatures = [c for c in opp_board if "Land" not in c["types"]]
        opp_lands = [c for c in opp_board if "Land" in c["types"]]

        hand = zones.get("my_hand", [])

        # Pre-render all images before touching the UI
        cand_highlights = {}
        for i in selected:
            cand_highlights[i] = "attack"

        opp_creature_imgs = self._pre_render_cards(opp_creatures)
        opp_land_imgs = self._pre_render_cards(opp_lands)
        my_creature_imgs = self._pre_render_cards(
            my_creatures, candidate_highlights=cand_highlights)
        my_land_imgs = self._pre_render_cards(my_lands)
        hand_imgs = self._pre_render_cards(hand)
        stack_imgs = self._pre_render_cards(
            zones.get("stack", []))
        priority_imgs = self._pre_render_priority(s)

        # Now do all UI updates in one batch
        self._apply_cards(self.opp_creatures, opp_creature_imgs, opp_creatures, new_photos, show_combat_tips=True)
        self._apply_cards(self.opp_lands, opp_land_imgs, opp_lands, new_photos)
        self._apply_cards(self.my_creatures, my_creature_imgs, my_creatures, new_photos, show_combat_tips=True)
        self._apply_cards(self.my_lands, my_land_imgs, my_lands, new_photos)
        self._apply_cards(self.hand_frame, hand_imgs, hand, new_photos)
        self._apply_cards(self.stack_frame, stack_imgs,
                          zones.get("stack", []), new_photos)
        self._apply_priority(self.priority_frame, priority_imgs, new_photos, s)

        # Swap photo references atomically
        self.card_photos = new_photos

        # Model prediction
        self._predict(s)

    def _pre_render_cards(self, cards, candidate_highlights=None):
        """Render card images off-screen, return list of PIL images."""
        if not HAS_PIL or not cards:
            return []
        images = []
        for i, info in enumerate(cards):
            hl = None
            if candidate_highlights and i in candidate_highlights:
                hl = candidate_highlights[i]
            images.append(draw_card_image(info, highlight=hl))
        return images

    def _pre_render_priority(self, s):
        """Pre-render candidate images for priority, target, card_select, mulligan."""
        dt = s.get("type")
        if not HAS_PIL:
            return None
        # For target/card_select/mulligan, render candidates as card images
        if dt in ("TARGET_SELECTION", "CARD_SELECTION", "MULLIGAN"):
            candidates = s.get("candidates", [])
            selected = set(s.get("selected", []))
            images = []
            for i, cf in enumerate(candidates):
                info = decode_card(cf)
                if not info:
                    images.append(None)
                    continue
                hl = "attack" if i in selected else None
                images.append(draw_card_image(info, highlight=hl))
            return images if images else None
        if dt != "PRIORITY_ACTION":
            return None
        candidates = s.get("candidates", [])
        selected = s.get("selected", [])
        sel_idx = selected[0] if selected else -1
        images = []
        for i, cf in enumerate(candidates):
            info = decode_action(cf)
            if not info:
                images.append(None)
                continue
            is_chosen = (i == sel_idx)
            is_pass = info.get("is_pass", False)
            images.append(draw_action_card_image(
                info, is_chosen=is_chosen, is_pass=is_pass))
        return images

    def _apply_cards(self, frame, images, cards, photos_out,
                     show_combat_tips=False):
        """Apply pre-rendered images to a frame, minimizing flicker."""
        for w in frame.winfo_children():
            w.destroy()

        if not cards:
            tk.Label(frame, text="(empty)", bg=frame["bg"],
                     fg="#585b70", font=("Consolas", 9)).pack(
                         side=tk.LEFT, padx=10, pady=20)
            return

        for i, info in enumerate(cards):
            if HAS_PIL and images and i < len(images) and images[i]:
                photo = ImageTk.PhotoImage(images[i])
                photos_out.append(photo)
                lbl = tk.Label(frame, image=photo, bg=frame["bg"])
                lbl.pack(side=tk.LEFT, padx=1, pady=2)
            else:
                c = info["colors"][0] if info["colors"] else "C"
                bg, fg = COLOR_HEX.get(c, COLOR_HEX["C"])
                text = info["label"]
                if info["power"] is not None:
                    text += f"\n{info['power']}/{info['toughness']}"
                lbl = tk.Label(frame, text=text, bg=bg, fg=fg,
                               font=("Consolas", 8), width=14, height=5,
                               relief="raised", borderwidth=2)
                lbl.pack(side=tk.LEFT, padx=1, pady=2)
            # Add combat math tooltip for battlefield creatures only
            if show_combat_tips:
                card_info = info
                CardTooltip(lbl, lambda ci=card_info: format_combat_tooltip(ci))

    def _apply_priority(self, frame, images, photos_out, s=None):
        """Apply pre-rendered priority images with optional tooltips."""
        for w in frame.winfo_children():
            w.destroy()

        if images is None:
            tk.Label(frame,
                     text="(no candidates)",
                     bg=frame["bg"],
                     fg="#585b70",
                     font=("Consolas", 9)).pack(
                         side=tk.LEFT, padx=10, pady=8)
            return

        # Decode candidates for tooltips (card-based decisions only)
        candidates = s.get("candidates", []) if s else []
        dt = s.get("type", "") if s else ""
        is_card_decision = dt in ("TARGET_SELECTION", "CARD_SELECTION",
                                   "MULLIGAN", "DECLARE_ATTACKERS",
                                   "DECLARE_BLOCKERS")

        for i, img in enumerate(images):
            if img and HAS_PIL:
                photo = ImageTk.PhotoImage(img)
                photos_out.append(photo)
                lbl = tk.Label(frame, image=photo,
                               bg=frame["bg"])
                lbl.pack(side=tk.LEFT, padx=1, pady=2)
                # Add tooltip for card-based candidates
                if is_card_decision and i < len(candidates):
                    info = decode_card(candidates[i])
                    if info:
                        CardTooltip(lbl, lambda ci=info: format_combat_tooltip(ci))

    def _update_eval_bar(self, value=None):
        """Draw a lichess-style vertical eval bar.
        value: float in [-1, 1], where +1 = player winning."""
        self.eval_canvas.update_idletasks()
        w = self.eval_canvas.winfo_width()
        h = self.eval_canvas.winfo_height()
        self.eval_canvas.delete("all")

        if value is None or h < 10:
            self.eval_canvas.create_rectangle(
                0, 0, w, h, fill="#555555", outline="")
            self.eval_label.config(text="--")
            self.eval_label_btm.config(text="--")
            return

        # win_pct: 0.0 (opp winning) to 1.0 (player winning)
        win_pct = (value + 1) / 2
        # Split point: opponent (top, black) vs player (bottom, white)
        split_y = int(h * (1 - win_pct))

        # Opponent portion (top) — dark
        if split_y > 0:
            self.eval_canvas.create_rectangle(
                0, 0, w, split_y, fill="#2b2b2b", outline="")
        # Player portion (bottom) — white
        if split_y < h:
            self.eval_canvas.create_rectangle(
                0, split_y, w, h, fill="#e8e8e8", outline="")

        # Center line
        mid = h // 2
        self.eval_canvas.create_line(
            0, mid, w, mid, fill="#666666", width=1, dash=(2, 2))

        # Labels
        pct = int(win_pct * 100)
        self.eval_label.config(text=f"{100-pct}%",
                                fg="#f38ba8")
        self.eval_label_btm.config(text=f"{pct}%",
                                    fg="#a6e3a1")

    def _predict(self, s):
        if self.mode == self.MODE_RL_REPLAY:
            self.pred_title_v.set("Current Model vs RL Policy")
        else:
            self.pred_title_v.set("Model vs Heuristic")

        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete("1.0", tk.END)

        if self.model is None:
            self.pred_text.insert("1.0", "No model loaded")
            self.pred_text.config(state=tk.DISABLED)
            self._update_eval_bar(None)
            return

        gf = s["global_features"]
        flat = s["game_state_flat"]

        g = np.zeros(GLOBAL_DIM, dtype=np.float32)
        gl = min(len(gf), GLOBAL_DIM)
        if gl > 0:
            g[:gl] = gf[:gl]
        np.clip(g, -10, 10, out=g)

        _, zones, masks = parse_game_state(flat, g)

        with torch.no_grad():
            t = lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)

            state = self.model.encode_state(
                t(g),
                t(zones["my_board"]), t(masks["my_board_mask"]),
                t(zones["opp_board"]), t(masks["opp_board_mask"]),
                t(zones["hand"]), t(masks["hand_mask"]),
                t(zones["my_gy"]), t(masks["my_gy_mask"]),
                t(zones["opp_gy"]), t(masks["opp_gy_mask"]),
                t(zones["stack"]), t(masks["stack_mask"]))

            value = self.model.get_value(state).item()
            self._last_value = value
            self._update_eval_bar(value)

            lines = []
            lines.append(f"Win probability: {(value+1)/2:.0%}")
            lines.append(f"Value: {value:+.3f}")
            lines.append("")
            correct = (value > 0) == s["won"]
            lines.append(f"Model: {'WIN' if value > 0 else 'LOSS'}")
            lines.append(f"Actual: {'WON' if s['won'] else 'LOST'}")
            lines.append(f"{'CORRECT' if correct else 'WRONG'}")
            lines.append("")

            candidates = s.get("candidates", [])
            dt = s.get("type", "")

            if candidates and dt == "DECLARE_ATTACKERS":
                n = len(candidates)
                cf_t = torch.zeros(1, n, CARD_DIM, device=self.device)
                cm_t = torch.ones(1, n, dtype=torch.bool, device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), CARD_DIM)
                    cf_t[0, j, :cl] = torch.tensor(cf[:cl], dtype=torch.float32)

                logits = self.model.attack_head(state, cf_t, cm_t)
                probs = torch.sigmoid(logits)

                lines.append("=== Attack Decision ===")
                lines.append("")
                for j in range(n):
                    p = probs[0, j].item()
                    info = decode_card(candidates[j])
                    sel = j in s["selected"]
                    marker = ">>" if sel else "  "
                    label = info["label"] if info else "?"
                    if info and info["power"] is not None:
                        label += f" {info['power']}/{info['toughness']}"
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}")
                    lines.append(f"   [{bar:20s}] {p:.0%}")
                    lines.append("")

                lines.append(f"{self._get_choice_label()}: {list(s['selected'])}")
                model_sel = [j for j in range(n) if probs[0,j].item() > 0.5]
                lines.append(f"{self._get_compare_label()}:     {model_sel}")
                if set(s["selected"]) == set(model_sel):
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            elif candidates and dt == "PRIORITY_ACTION":
                n = len(candidates)
                af_t = torch.zeros(
                    1, n, 64, device=self.device)
                am_t = torch.zeros(
                    1, n, dtype=torch.bool,
                    device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), 64)
                    af_t[0, j, :cl] = torch.tensor(
                        cf[:cl], dtype=torch.float32)
                    am_t[0, j] = True

                logits = self.model.priority_head(
                    state, af_t, am_t)
                probs = torch.softmax(logits, dim=-1)

                lines.append("=== Priority Decision ===")
                lines.append("")
                sel_idx = s["selected"][0] \
                    if s["selected"] else -1
                for j in range(n):
                    p = probs[0, j].item()
                    info = decode_action(candidates[j])
                    sel = (j == sel_idx)
                    marker = ">>" if sel else "  "
                    label = info["label"] if info else "?"
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}")
                    lines.append(
                        f"   [{bar:20s}] {p:.0%}")
                    lines.append("")

                model_pick = probs[0].argmax().item()
                model_info = decode_action(
                    candidates[model_pick])
                model_label = model_info["label"] \
                    if model_info else "?"
                heur_info = decode_action(
                    candidates[sel_idx]) if sel_idx >= 0 \
                    else None
                heur_label = heur_info["label"] \
                    if heur_info else "PASS"
                lines.append(
                    f"{self._get_choice_label()}: [{sel_idx}] {heur_label}")
                lines.append(
                    f"{self._get_compare_label()}:     [{model_pick}] "
                    f"{model_label}")
                if model_pick == sel_idx:
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            elif candidates and dt == "DECLARE_BLOCKERS":
                # Candidates are (blocker, attacker) concatenated pairs
                # Last candidate is "no block" zero vector
                real_pairs = candidates[:-1] if len(candidates) > 1 else candidates
                n_pairs = len(real_pairs)

                if n_pairs > 0:
                    card_dim = CARD_DIM
                    # Infer n_attackers: consecutive pairs with same blocker features
                    first_blocker = np.array(real_pairs[0][:card_dim])
                    n_attackers = 1
                    for j in range(1, n_pairs):
                        other = np.array(real_pairs[j][:card_dim])
                        if np.allclose(first_blocker, other, atol=0.01):
                            n_attackers += 1
                        else:
                            break
                    n_blockers = n_pairs // max(n_attackers, 1)

                    # Build separate blocker/attacker tensors
                    bf = torch.zeros(1, n_blockers, card_dim, device=self.device)
                    bm = torch.ones(1, n_blockers, dtype=torch.bool, device=self.device)
                    af = torch.zeros(1, n_attackers, card_dim, device=self.device)
                    am = torch.ones(1, n_attackers, dtype=torch.bool, device=self.device)

                    for b in range(n_blockers):
                        pair_idx = b * n_attackers
                        if pair_idx < n_pairs:
                            feats = real_pairs[pair_idx]
                            bf[0, b, :min(card_dim, len(feats))] = \
                                torch.tensor(feats[:card_dim], dtype=torch.float32)
                    for a in range(n_attackers):
                        if a < n_pairs:
                            feats = real_pairs[a]
                            af[0, a, :min(card_dim, len(feats)-card_dim)] = \
                                torch.tensor(feats[card_dim:card_dim*2], dtype=torch.float32)

                    # BlockHead: (batch, n_blockers, n_attackers+1)
                    logits = self.model.block_head(
                        state, bf, bm, af, am)

                    lines.append("=== Block Decision ===")
                    lines.append(f"({n_blockers} blockers × {n_attackers} attackers)")
                    lines.append("")

                    for b in range(n_blockers):
                        probs_b = torch.softmax(logits[0, b], dim=-1)
                        blocker_info = decode_card(real_pairs[b * n_attackers][:card_dim])
                        b_label = blocker_info["label"] if blocker_info else "?"
                        if blocker_info and blocker_info["power"] is not None:
                            b_label += f" {blocker_info['power']}/{blocker_info['toughness']}"
                        lines.append(f"Blocker: {b_label}")

                        best_a = probs_b.argmax().item()
                        for a in range(n_attackers):
                            pair_idx = b * n_attackers + a
                            p = probs_b[a].item()
                            atk_info = decode_card(real_pairs[a][card_dim:card_dim*2])
                            a_label = atk_info["label"] if atk_info else "?"
                            if atk_info and atk_info["power"] is not None:
                                a_label += f" {atk_info['power']}/{atk_info['toughness']}"
                            sel = pair_idx in s["selected"]
                            marker = ">>" if sel else "  "
                            pick = " <MODEL" if a == best_a and best_a < n_attackers else ""
                            bar = "#" * int(p * 20)
                            lines.append(f"  {marker} → {a_label}{pick}")
                            lines.append(f"     [{bar:20s}] {p:.0%}")

                        # "Don't block" option
                        no_block_p = probs_b[-1].item()
                        pick = " <MODEL" if best_a == n_attackers else ""
                        bar = "#" * int(no_block_p * 20)
                        lines.append(f"     → Don't block{pick}")
                        lines.append(f"     [{bar:20s}] {no_block_p:.0%}")
                        lines.append("")

                    # Model picks
                    model_sel = []
                    for b in range(n_blockers):
                        probs_b = torch.softmax(logits[0, b], dim=-1)
                        best_a = probs_b.argmax().item()
                        if best_a < n_attackers:
                            model_sel.append(b * n_attackers + best_a)

                    lines.append(f"{self._get_choice_label()}: {list(s['selected'])}")
                    lines.append(f"{self._get_compare_label()}:     {model_sel}")
                    if set(s["selected"]) == set(model_sel):
                        lines.append("MATCH!")
                    else:
                        lines.append("DIFFERENT")

            elif dt == "TARGET_SELECTION" and candidates:
                n = len(candidates)
                cf_t = torch.zeros(1, n, CARD_DIM, device=self.device)
                cm_t = torch.ones(1, n, dtype=torch.bool, device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), CARD_DIM)
                    cf_t[0, j, :cl] = torch.tensor(cf[:cl], dtype=torch.float32)

                logits = self.model.target_head(state, cf_t, cm_t)
                probs = torch.softmax(logits, dim=-1)

                lines.append("=== Target Decision ===")
                lines.append(f"Context: {s.get('info', '')}")
                lines.append("")
                sel_indices = set(s.get("selected", []))
                model_pick = probs[0].argmax().item()
                for j, cf in enumerate(candidates):
                    p = probs[0, j].item()
                    is_player = len(cf) >= 5 and cf[4] > 0.5
                    if is_player:
                        life = int(cf[0] * 50 - 10)
                        label = f"Player (life={life})"
                    else:
                        info = decode_card(cf)
                        label = info["label"] if info else "?"
                        if info and info["power"] is not None:
                            label += f" {info['power']}/{info['toughness']}"
                    sel = j in sel_indices
                    marker = ">>" if sel else "  "
                    pick = " <MODEL" if j == model_pick else ""
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}{pick}")
                    lines.append(f"   [{bar:20s}] {p:.0%}")
                lines.append("")
                sel_idx = list(sel_indices)[0] if sel_indices else -1
                lines.append(f"{self._get_choice_label()}: {list(s['selected'])}")
                lines.append(f"{self._get_compare_label()}: [{model_pick}]")
                if model_pick == sel_idx:
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            elif dt == "CARD_SELECTION" and candidates:
                n = len(candidates)
                cf_t = torch.zeros(1, n, CARD_DIM, device=self.device)
                cm_t = torch.ones(1, n, dtype=torch.bool, device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), CARD_DIM)
                    cf_t[0, j, :cl] = torch.tensor(cf[:cl], dtype=torch.float32)

                logits = self.model.card_select_head(state, cf_t, cm_t)
                probs = torch.sigmoid(logits)

                lines.append("=== Card Selection ===")
                lines.append(f"Context: {s.get('info', '')}")
                lines.append("")
                sel_indices = set(s.get("selected", []))
                for j, cf in enumerate(candidates):
                    p = probs[0, j].item()
                    info = decode_card(cf)
                    label = info["label"] if info else "?"
                    if info and info["power"] is not None:
                        label += f" {info['power']}/{info['toughness']}"
                    sel = j in sel_indices
                    marker = ">>" if sel else "  "
                    bar = "#" * int(p * 20)
                    lines.append(f"{marker} {label}")
                    lines.append(f"   [{bar:20s}] {p:.0%}")
                lines.append("")
                lines.append(f"{self._get_choice_label()}: {list(s['selected'])}")
                model_sel = [j for j in range(n) if probs[0,j].item() > 0.5]
                lines.append(f"{self._get_compare_label()}: {model_sel}")

            elif dt == "MULLIGAN" and candidates:
                n = len(candidates)
                hf_t = torch.zeros(1, n, CARD_DIM, device=self.device)
                hm_t = torch.ones(1, n, dtype=torch.bool, device=self.device)
                for j, cf in enumerate(candidates):
                    cl = min(len(cf), CARD_DIM)
                    hf_t[0, j, :cl] = torch.tensor(cf[:cl], dtype=torch.float32)

                keep_logit = self.model.mulligan_head(state, hf_t, hm_t)
                keep_prob = torch.sigmoid(keep_logit).item()

                sel = s.get("selected", [])
                keep = sel and sel[0] == 1
                model_keep = keep_prob > 0.5

                lines.append("=== Mulligan Decision ===")
                lines.append(f"Context: {s.get('info', '')}")
                lines.append("")
                bar = "#" * int(keep_prob * 20)
                lines.append(f"Keep probability:")
                lines.append(f"   [{bar:20s}] {keep_prob:.0%}")
                lines.append("")
                lines.append(f"{self._get_choice_label()}: {'KEEP' if keep else 'MULLIGAN'}")
                lines.append(f"{self._get_compare_label()}: {'KEEP' if model_keep else 'MULLIGAN'}")
                if keep == model_keep:
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")
                lines.append("")
                lines.append("Hand:")
                for j, cf in enumerate(candidates):
                    info = decode_card(cf)
                    label = info["label"] if info else "?"
                    if info and info.get("cmc", 0) > 0:
                        label += f" [{info['cmc']}]"
                    lines.append(f"  {label}")

            elif dt == "BINARY_CHOICE":
                logit = self.model.binary_head(state)
                prob = torch.sigmoid(logit).item()

                sel = s.get("selected", [])
                yes = sel and sel[0] == 1
                model_yes = prob > 0.5

                lines.append("=== Binary Decision ===")
                lines.append(f"Context: {s.get('info', '')}")
                lines.append("")
                bar = "#" * int(prob * 20)
                lines.append(f"Yes probability:")
                lines.append(f"   [{bar:20s}] {prob:.0%}")
                lines.append("")
                lines.append(f"{self._get_choice_label()}: {'YES' if yes else 'NO'}")
                lines.append(f"{self._get_compare_label()}: {'YES' if model_yes else 'NO'}")
                if yes == model_yes:
                    lines.append("MATCH!")
                else:
                    lines.append("DIFFERENT")

            # Append global combat overview
            gc = decode_global_combat(gf)
            if gc:
                lines.append("")
                lines.append("")
                lines.append(format_global_combat(gc))

            self.pred_text.insert("1.0", "\n".join(lines))
        self.pred_text.config(state=tk.DISABLED)


def load_replay_runs(trace_dir, replay_id):
    runs = {}
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return runs

    pattern = f"replay_{replay_id}_*.jsonl"
    for path in trace_path.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue
        if not lines:
            continue
        header = lines[0]
        if header.get("recordType") != "header":
            continue
        key = (header.get("runLabel"), int(header.get("seatIndex", 1)))
        runs[key] = {
            "header": header,
            "steps": [line for line in lines[1:]
                      if line.get("recordType") == "step"],
            "path": str(path),
        }
    return runs


def summarize_zone_cards(zone, mask, limit=8):
    cards = []
    for row, present in zip(zone, mask):
        if not present:
            continue
        card = decode_card(row)
        if card is None:
            continue
        label = "/".join(card.get("types", [])) or "Card"
        if card.get("power") is not None and card.get("toughness") is not None:
            label += f" {card['power']}/{card['toughness']}"
        if card.get("tapped"):
            label += " tapped"
        cards.append(label)
        if len(cards) >= limit:
            break
    return cards


def decode_snapshot_globals(global_features):
    if len(global_features) < 2:
        return None
    return {
        "my_life": int(round(global_features[0] * 50 - 10)),
        "opp_life": int(round(global_features[1] * 50 - 10)),
    }


def format_snapshot_card(card):
    if not card:
        return "Card"
    label = card.get("name") or "Card"
    type_line = card.get("type")
    basic_land_suffix = f"Basic Land - {label}"
    if type_line and type_line != basic_land_suffix:
        label += f" ({type_line})"
    power = card.get("power")
    toughness = card.get("toughness")
    if power is not None and toughness is not None:
        label += f" {power}/{toughness}"
    flags = []
    if card.get("tapped"):
        flags.append("tapped")
    if card.get("summoningSick"):
        flags.append("summoning sick")
    counters = card.get("counters")
    if counters and counters != "{}":
        flags.append(f"counters={counters}")
    if flags:
        label += " [" + ", ".join(flags) + "]"
    return label


def summarize_snapshot_payload(snapshot):
    if not snapshot:
        return None

    sections = []
    overview = [
        f"My life: {snapshot.get('myLife', '?')} | Opp life: {snapshot.get('oppLife', '?')}",
        f"My poison: {snapshot.get('myPoison', 0)} | Opp poison: {snapshot.get('oppPoison', 0)}",
        f"My hand: {snapshot.get('myHandCount', 0)} | Opp hand: {snapshot.get('oppHandCount', 0)}",
        f"My library: {snapshot.get('myLibraryCount', 0)} | Opp library: {snapshot.get('oppLibraryCount', 0)}",
    ]
    active = snapshot.get("activePlayer")
    if active:
        overview.append(f"Active player: {active}")
    sections.append("\n".join(overview))

    zone_specs = [
        ("myBoard", "My board"),
        ("oppBoard", "Opp board"),
        ("hand", "Hand"),
        ("myGraveyard", "My gy"),
        ("oppGraveyard", "Opp gy"),
    ]
    for key, label in zone_specs:
        cards = snapshot.get(key) or []
        if cards:
            sections.append(f"{label}: " + ", ".join(format_snapshot_card(card) for card in cards))
        else:
            sections.append(f"{label}: empty")

    stack_items = snapshot.get("stack") or []
    if stack_items:
        sections.append("Stack: " + ", ".join(stack_items))
    else:
        sections.append("Stack: empty")
    return "\n\n".join(sections)


def summarize_snapshot(global_features, game_state_flat, snapshot=None):
    rich_snapshot = summarize_snapshot_payload(snapshot)
    if rich_snapshot:
        return rich_snapshot

    try:
        gf = np.array(global_features, dtype=np.float32)
        flat = np.array(game_state_flat, dtype=np.float32)
        _, zones, masks = parse_game_state(flat, gf)
    except Exception:
        return "Snapshot unavailable"

    sections = []
    snapshot_globals = decode_snapshot_globals(gf)
    if snapshot_globals:
        sections.append(
            f"My life: {snapshot_globals['my_life']} | Opp life: {snapshot_globals['opp_life']}"
        )
    zone_order = [
        ("my_board", "My board"),
        ("opp_board", "Opp board"),
        ("hand", "Hand"),
        ("my_gy", "My gy"),
        ("opp_gy", "Opp gy"),
        ("stack", "Stack"),
    ]
    for key, label in zone_order:
        cards = summarize_zone_cards(zones[key], masks[f"{key}_mask"])
        if cards:
            sections.append(f"{label}: " + ", ".join(cards))
        else:
            sections.append(f"{label}: empty")
    return "\n\n".join(sections)


def first_divergence_index(left_steps, right_steps):
    limit = min(len(left_steps), len(right_steps))
    for idx in range(limit):
        if left_steps[idx].get("decisionHash") != right_steps[idx].get("decisionHash"):
            return idx
    if len(left_steps) != len(right_steps):
        return limit
    return None


class ReplayDiffViewer:
    def __init__(self, root, trace_dir, replay_id, seat):
        self.root = root
        self.root.title(f"Replay Diff - {replay_id}")
        runs = load_replay_runs(trace_dir, replay_id)
        if not runs:
            raise SystemExit(f"No replay traces found for replay_id={replay_id}")

        self.left = runs.get(("baseline", seat))
        self.right = runs.get(("compare", seat))
        if self.left is None or self.right is None:
            keys = sorted(runs.keys())
            if len(keys) >= 2:
                self.left = runs[keys[0]]
                self.right = runs[keys[1]]
            else:
                raise SystemExit("Need two replay traces to compare")

        self.left_steps = self.left["steps"]
        self.right_steps = self.right["steps"]
        self.max_idx = max(len(self.left_steps), len(self.right_steps)) - 1
        self.idx = 0
        self.divergence = first_divergence_index(self.left_steps, self.right_steps)
        self._syncing_controls = False

        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Prev", command=self.prev_step).pack(side="left")
        ttk.Button(top, text="Next", command=self.next_step).pack(side="left")
        ttk.Button(top, text="Jump Divergence",
                   command=self.jump_divergence).pack(side="left", padx=6)
        ttk.Label(top, text="Go to step:").pack(side="left", padx=(8, 2))
        self.step_var = tk.StringVar(value="1")
        step_entry = ttk.Entry(top, textvariable=self.step_var, width=8)
        step_entry.pack(side="left")
        step_entry.bind("<Return>", lambda _event: self.jump_to_step())
        ttk.Button(top, text="Go", command=self.jump_to_step).pack(side="left", padx=(4, 8))
        self.status = tk.StringVar()
        ttk.Label(top, textvariable=self.status).pack(side="left", padx=10)

        scrub = ttk.Frame(root)
        scrub.pack(fill="x", padx=8, pady=(0, 8))
        self.step_scale = tk.Scale(
            scrub,
            from_=1,
            to=max(1, self.max_idx + 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
            resolution=1,
            command=self.on_scale_change,
        )
        self.step_scale.pack(fill="x")

        panes = ttk.Frame(root)
        panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.left_text = tk.Text(panes, width=80, height=40, wrap="word")
        self.right_text = tk.Text(panes, width=80, height=40, wrap="word")
        self.left_text.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.right_text.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.render()

    def prev_step(self):
        self.idx = max(0, self.idx - 1)
        self.render()

    def next_step(self):
        self.idx = min(self.max_idx, self.idx + 1)
        self.render()

    def jump_divergence(self):
        if self.divergence is not None:
            self.idx = self.divergence
            self.render()

    def jump_to_step(self):
        try:
            requested = int(self.step_var.get())
        except (TypeError, ValueError):
            requested = self.idx + 1
        self.idx = max(0, min(self.max_idx, requested - 1))
        self.render()

    def on_scale_change(self, value):
        if self._syncing_controls:
            return
        try:
            requested = int(float(value))
        except (TypeError, ValueError):
            return
        self.idx = max(0, min(self.max_idx, requested - 1))
        self.render()

    def render(self):
        left = self.left_steps[self.idx] if self.idx < len(self.left_steps) else None
        right = self.right_steps[self.idx] if self.idx < len(self.right_steps) else None
        hash_match = (left and right
                      and left.get("stateHash") == right.get("stateHash")
                      and left.get("decisionHash") == right.get("decisionHash"))
        badge = "MATCH" if hash_match else "DIFF"
        self.status.set(
            f"Step {self.idx + 1}/{self.max_idx + 1} | First divergence: "
            f"{'none' if self.divergence is None else self.divergence + 1} | {badge}"
        )
        self._syncing_controls = True
        self.step_var.set(str(self.idx + 1))
        self.step_scale.set(self.idx + 1)
        self._syncing_controls = False
        self._fill_text(self.left_text, self._render_step("Baseline", self.left["header"], left))
        self._fill_text(self.right_text, self._render_step("Compare", self.right["header"], right))

    def _fill_text(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.config(state=tk.DISABLED)

    def _render_step(self, title, header, step):
        lines = [
            title,
            f"Actor: {header.get('actor')} (seat {header.get('seatIndex')})",
            f"Policy: {header.get('policyMode')} via {header.get('backend')}",
            f"Model: {header.get('modelId')}",
            f"Deck: {header.get('playerDeck')}",
            f"Opponent Deck: {header.get('opponentDeck')}",
            "",
        ]
        if step is None:
            lines.append("No step at this index")
            return "\n".join(lines)

        lines.extend([
            f"Turn {step.get('turn')} | Phase {step.get('phase')}",
            f"Decision: {step.get('decisionType')}",
            f"Context: {step.get('contextInfo')}",
            f"Selected: {', '.join(step.get('selectedLabels', [])) or step.get('selectedIndices')}",
            f"Value estimate: {self._format_value(step.get('valueEstimate'))}",
            f"State hash: {step.get('stateHash')}",
            f"Decision hash: {step.get('decisionHash')}",
            "",
            "Candidates:",
        ])
        candidate_labels = step.get("traceCandidateLabels") or step.get("candidateLabels", [])
        selected = set(step.get("selectedIndices", []))
        for idx, label in enumerate(candidate_labels[:30]):
            marker = "*" if idx in selected else " "
            lines.append(f"{marker} [{idx}] {label}")
        if len(candidate_labels) > 30:
            lines.append(f"... {len(candidate_labels) - 30} more")
        missing_selected = [
            label for label in (step.get("selectedLabels", []) or [])
            if label not in candidate_labels
        ]
        if missing_selected:
            lines.append("")
            lines.append("Selected outside raw candidates:")
            for label in missing_selected:
                lines.append(f"* {label}")
        lines.extend([
            "",
            "Snapshot:",
            summarize_snapshot(step.get("globalFeatures", []),
                               step.get("gameStateFlat", []),
                               step.get("snapshot")),
        ])
        return "\n".join(lines)

    @staticmethod
    def _format_value(value):
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
        default="../../rl_data/trajectories")
    parser.add_argument("--replay-trace-dir",
        default=None)
    parser.add_argument("--replay-id",
        default=None)
    parser.add_argument("--seat", type=int,
        default=1)
    parser.add_argument("--model",
        default="../../rl_data/checkpoints/"
                "model_with_decisions.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int,
        default=500)
    args = parser.parse_args()
    if args.replay_trace_dir and args.replay_id:
        root = tk.Tk()
        ReplayDiffViewer(root, args.replay_trace_dir,
                         args.replay_id, args.seat)
        root.mainloop()
        return

    # Load both PPO and heuristic data — mode toggle switches between them
    print("Loading samples...", flush=True)
    samples = []
    base_dir = os.path.dirname(args.data_dir)

    # PPO trajectories (RL decisions)
    ppo_eval_dir = os.path.join(base_dir, 'ppo_trajectories_eval')
    ppo_dir = os.path.join(base_dir, 'ppo_trajectories')
    for d, label in [(ppo_eval_dir, "PPO eval"),
                     (ppo_dir, "PPO collect")]:
        if os.path.isdir(d):
            ppo_samples = load_samples(
                d, args.max_samples, rl_only=True,
                source_tag="ppo")
            if ppo_samples:
                samples.extend(ppo_samples)
                print(f"  + {len(ppo_samples)} {label} "
                      f"samples (RL replay)", flush=True)

    # Base trajectories (heuristic decisions)
    heur_samples = load_samples(args.data_dir,
                                args.max_samples,
                                source_tag="heuristic")
    if heur_samples:
        samples.extend(heur_samples)
        print(f"  + {len(heur_samples)} heuristic "
              f"samples", flush=True)

    n_ppo = sum(1 for s in samples if s.get("source") == "ppo")
    n_heur = sum(1 for s in samples if s.get("source") == "heuristic")
    from collections import Counter
    type_counts = Counter(s["type"] for s in samples)
    print(f"  {len(samples)} total ({n_ppo} RL replay, "
          f"{n_heur} heuristic)", flush=True)
    for dt, c in type_counts.most_common():
        print(f"    {dt}: {c}", flush=True)

    model_path = args.model
    model = None
    default_model = ('../../rl_data/checkpoints/'
                     'model_with_decisions.pt')
    explicitly_set = (
        args.model != default_model and
        not args.model.endswith(
            'model_with_decisions.pt') and
        not args.model.endswith(
            'best_value_model.pt'))
    if explicitly_set:
        # User specified a model — use exactly that
        candidates = [model_path]
    else:
        # Auto-detect: try best available in order of preference
        ckpt_dir = os.path.dirname(model_path)
        candidates = [
            os.path.join(ckpt_dir, 'best_ppo_model.pt'),
            os.path.join(ckpt_dir, 'model_with_decisions.pt'),
            os.path.join(ckpt_dir, 'best_priority_model.pt'),
            os.path.join(ckpt_dir, 'best_attack_model.pt'),
            os.path.join(ckpt_dir, 'best_block_model.pt'),
            model_path,
        ]
    for p in candidates:
        if p and os.path.exists(p):
            print(f"Loading model: {p}", flush=True)
            backend = resolve_backend(args.device)
            model = MTGModel.load(p, device=backend.name)
            model.eval()
            model_path = p
            break

    root = tk.Tk()
    GameStateViewer(root, samples, model, backend.torch_device if model is not None else resolve_backend(args.device).torch_device,
                    model_path=model_path,
                    data_dir=args.data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
