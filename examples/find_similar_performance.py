import json


def get_other_performances(perf_id, metadata):
    for work_id, perfs in metadata.items():
        if perf_id in perfs:
            others = {pid: perfs[pid] for pid in perfs if pid != perf_id}
            return work_id, others
    return None, {}

with open("../data/da-tacos/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json") as f:
    metadata = json.load(f)

perf_id = "P_747569"

found = False
fround_work_id = None
for work_id, perfs in metadata.items():
    if perf_id in perfs:
        print("Found in work:", work_id)
        found = True
        fround_work_id = work_id
        break

print("Found?" , found)
print(metadata[fround_work_id].keys())

# Example
perf_id = "P_747569"
work_id, others = get_other_performances(perf_id, metadata)

if work_id:
    print(f"{perf_id} belongs to work {work_id}")
    if others:
        print("Other performances of the same work:")
        for pid, meta in others.items():
            print(f"  {pid} by {meta.get('perf_artist','Unknown')} — {meta.get('perf_title','')}")
    else:
        print("No other performances (unique recording).")
else:
    print(f"{perf_id} not found in da-TACOS metadata.")

