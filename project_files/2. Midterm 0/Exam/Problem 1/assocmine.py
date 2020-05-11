#!/usr/bin/env python3

from collections import defaultdict

def make_itemsets(item_lists):
    return [set(l) for l in item_lists]

def update_item_counts(item_counts, itemset):
    for a in itemset:
        item_counts[a] += 1

def count_items(receipts):
    item_counts = defaultdict(int) # item -> count
    for itemset in receipts:
        update_item_counts(item_counts, itemset)
    return item_counts

def get_rare_items(item_counts, min_item_count):
    rare_items = set()
    for item, count in item_counts.items():
        if count < min_item_count:
            rare_items |= {item}
    return rare_items

def find_assoc_rules(receipts, threshold, min_item_count=0):
    item_counts = count_items(receipts)
    rare_items = get_rare_items(item_counts, min_item_count)

    # Count pairs for non-rare items
    pair_counts = defaultdict(int) # (item_a, item_b) -> count
    for itemset in receipts:
        update_pair_counts(pair_counts, itemset - rare_items)

    # Compute and return the rules
    return filter_rules_by_conf(pair_counts, item_counts, threshold)

def filter_rules_by_conf(pair_counts, item_counts, threshold):
    rules = {} # (item_a, item_b) -> conf (item_a => item_b)
    for (a, b) in pair_counts:
        assert a in item_counts
        conf_ab = pair_counts[(a, b)] / item_counts[a]
        if conf_ab >= threshold:
            rules[(a, b)] = conf_ab
    return rules

def update_pair_counts (pair_counts, itemset):
    """
    Updates a dictionary of pair counts for
    all pairs of items in a given itemset.
    """
    assert type (pair_counts) is defaultdict
    from itertools import combinations
    for (a, b) in combinations(itemset, 2):
        pair_counts[(a, b)] += 1
        pair_counts[(b, a)] += 1

def gen_rule_str(a, b, val=None, val_fmt='{:.3f}', sep=" = ", prefix=""):
    text = "{} => {}".format(a, b)
    if val:
        text = prefix + "conf(" + text + ")"
        text += sep + val_fmt.format(val)
    return text

def print_rules(rules, rank=None, **kwargs):
    if type(rules) is dict or type(rules) is defaultdict:
        from operator import itemgetter
        ordered_rules = sorted(rules.items(), key=itemgetter(1), reverse=True)
    else: # Assume rules is iterable
        ordered_rules = [((a, b), None) for a, b in rules]
    if rank is not None:
        ordered_rules = ordered_rules[:rank]
    for (a, b), conf_ab in ordered_rules:
        print(gen_rule_str(a, b, conf_ab, **kwargs))

# eof
