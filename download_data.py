import os
from dotenv import load_dotenv
from datasets import load_dataset

def load_and_save_datasets():
    load_dotenv()
    my_token = os.getenv("AUTH_TOKEN")

    langs = ["eng", "nld", "zho"]

    os.makedirs("data", exist_ok=True)

    for l in langs:
        dataset = load_dataset(f"BabyLM-community/babylm-{l}", split="train", token=my_token)
        
        file_path = f"data/babylm-{l}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            for row in dataset:
                f.write(row['text'] + '\n')
        
        print(f"Finished! Saved to {file_path}")

if __name__ == "__main__":
    load_and_save_datasets()