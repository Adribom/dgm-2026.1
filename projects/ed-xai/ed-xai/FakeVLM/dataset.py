# Adapted from FakeVLM/datasets.py — simplified for image-only FakeClue.

import os
import json
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: Optional[str] = None,
        data: Optional[List] = None,
        image_folder: Optional[str] = None,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ) -> None:
        super().__init__()
        if data is not None:
            self.list_data_dict = data
        else:
            with open(data_path, "r") as f:
                self.list_data_dict = json.load(f)
        self.image_folder = image_folder
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.is_text_only = [
            "image" not in source for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]

        images = []
        if "image" in source:
            image_sources = source["image"]
            if isinstance(image_sources, str):
                image_sources = [image_sources]
            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                images.append(Image.open(image_path).convert("RGB"))

        system_prompt = source.get("system_prompt")

        convs = []
        assert len(source["conversations"]) > 0
        for j, conv in enumerate(source["conversations"]):
            expected = self.user_key if j % 2 == 0 else self.assistant_key
            assert conv["from"] == expected, "Invalid conversation turn order"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversation turns"

        return dict(
            images=images,
            videos=[],
            conversations=convs,
            system_prompt=system_prompt,
        )
