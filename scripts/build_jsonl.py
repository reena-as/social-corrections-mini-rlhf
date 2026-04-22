#!/usr/bin/env python
"""Convert data/correction_dataset.json -> data/processed/{train,val,test}_{sft,dpo}.jsonl"""
from social_corrections.data.build_jsonl import main

if __name__ == "__main__":
    main()
