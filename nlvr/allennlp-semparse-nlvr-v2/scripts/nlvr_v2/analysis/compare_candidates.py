import json

file1 = "../../nfs2_nitishg/data/nlvr/processed/agenda_v5_partial_False/search_candidates.txt"
file2 = "../../nfs2_nitishg/data/nlvr/processed/agenda_v6_partial_False/search_candidates.txt"

with open(file1) as f:
    candidates_v2 = json.load(f)

with open(file2) as f:
    candidates_v4 = json.load(f)


id2candidates_v2 = {d["id"]: d for d in candidates_v2}
id2candidates_v4 = {d["id"]: d for d in candidates_v4}

ids2 = set(id2candidates_v2.keys())
ids4 = set(id2candidates_v4.keys())

print(f"num_examples_1: {len(ids2)}  num_examples_2: {len(ids4)}")

# Extra in v2
v2_extra = ids2.difference(ids4)
print("num-ex extra in candidates_1: {}".format(len(v2_extra)))
v4_extra = ids4.difference(ids2)
print("num-ex extra in candidates_2: {}".format(len(v4_extra)))


# for i in v4_extra:
#     u = id2candidates_v4[i]["sentence"]
#     print(json.dumps(id2candidates_v4[i], indent=2))

for i in v2_extra:
    u = id2candidates_v2[i]["sentence"]
    print(json.dumps(id2candidates_v2[i], indent=2))

in_both = 0
diff_candidates = 0
for i, d4 in id2candidates_v4.items():
    if i in id2candidates_v2:
        in_both += 1
        d2 = id2candidates_v2[i]
        v4_candidates = d4["candidate_logical_forms"]
        v2_candidates = d2["candidate_logical_forms"]
        if v4_candidates != v2_candidates:
            # print(d2["sentence"])
            # print("\n".join(d4["candidate_logical_forms"]))
            # print("----------")
            # print("\n".join(d2["candidate_logical_forms"]))
            # print()
            diff_candidates += 1

print("Common ex in both: {}".format(in_both))
print("Examples with different candidates: {}".format(diff_candidates))
