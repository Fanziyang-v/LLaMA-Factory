import os
import json
from datasets import load_dataset, get_dataset_config_names

dataset = "open-r1/Mixture-of-Thoughts"


configs = get_dataset_config_names(dataset)
print(f"all configs: {configs}")
# ['all', 'code', 'math', 'science']


prefix = "open-r1--mixture-of-thoughts"

configs = ("code", "math")

for cfg in configs:
    output_json_file = f"{prefix}-{cfg}.json"
    all_data = []

    # load dataset.
    ds = load_dataset(dataset, cfg)["train"]
    print(f"Number of examples in config: {cfg} is {len(ds)}.")

    for data in ds:
        messages = data["messages"]
        num_tokens = data["num_tokens"]
        source = data["source"]
        # Init item.
        item = {
            "conversations": [],
            "num_tokens": num_tokens,
            "source": source,
        }

        assert len(messages) % 2 == 0
        num_rounds = len(messages) // 2
        for i in range(num_rounds):
            query_data, response_data = messages[2 * i : 2 * (i + 1)]
            assert query_data["role"] == "user"
            assert response_data["role"] == "assistant"
            item["conversations"].append(
                {
                    "from": query_data["role"],
                    "value": query_data["content"],
                }
            )
            item["conversations"].append(
                {
                    "from": response_data["role"],
                    "value": response_data["content"],
                }
            )
        all_data.append(item)
    # Save json file.
    with open(output_json_file, "w") as f:
        f.write(json.dumps(all_data, indent=2))
