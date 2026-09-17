"""
Federated Dataset Pipeline for Indian Sign Language (ISL) Recognition.
Combines 5 structurally diverse ISL datasets into a unified PyTorch ConcatDataset:
  1. Bridge Connectivity Sign Dictionary (WebDataset .tar streaming / pose-mediapipe)
  2. INCLUDE Dataset (Nested directories .MOV/.mp4 + MediaPipe)
  3. CISLR Corpus (Raw .mp4 + dataset.csv mapping)
  4. Mendeley Data ISL (.mp4 clips + custom JSON map)
  5. Kaggle ISL Datasets (.mp4 temporal clips + custom JSON map)

All streams output tensor shape: [30, 258] (30 frames, 258 spatial coordinates).
"""

import os
import sys
import csv
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from training.preprocessing.mediapipe_utils import (
    MediaPipeFeatureExtractor,
    temporal_resample_or_pad
)


class MasterGlossMap:
    """
    Unified Gloss Vocabulary Dictionary.
    Maps disparate sign labels from all dataset sources to a contiguous set of integer IDs.
    """
    def __init__(self, initial_glosses: Optional[List[str]] = None):
        self.gloss_to_id: Dict[str, int] = {}
        self.id_to_gloss: Dict[int, str] = {}
        
        if initial_glosses:
            for g in initial_glosses:
                self.get_or_add(g)

    def _normalize(self, gloss: str) -> str:
        """Normalizes gloss strings (lowercase, stripped, removes extra punctuation)."""
        return str(gloss).strip().lower().replace("-", "_").replace(" ", "_")

    def get_or_add(self, gloss: str) -> int:
        norm_gloss = self._normalize(gloss)
        if norm_gloss not in self.gloss_to_id:
            idx = len(self.gloss_to_id)
            self.gloss_to_id[norm_gloss] = idx
            self.id_to_gloss[idx] = norm_gloss
        return self.gloss_to_id[norm_gloss]

    def get_id(self, gloss: str) -> int:
        norm_gloss = self._normalize(gloss)
        if norm_gloss not in self.gloss_to_id:
            raise KeyError(f"Gloss '{gloss}' (normalized '{norm_gloss}') not in MasterGlossMap.")
        return self.gloss_to_id[norm_gloss]

    def get_gloss(self, idx: int) -> str:
        return self.id_to_gloss.get(idx, "<UNK>")

    def save(self, json_path: str):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "gloss_to_id": self.gloss_to_id,
                "id_to_gloss": {str(k): v for k, v in self.id_to_gloss.items()}
            }, f, indent=2)

    @classmethod
    def load(cls, json_path: str) -> "MasterGlossMap":
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        instance = cls()
        instance.gloss_to_id = data["gloss_to_id"]
        instance.id_to_gloss = {int(k): v for k, v in data["id_to_gloss"].items()}
        return instance

    def __len__(self) -> int:
        return len(self.gloss_to_id)


# ============================================================================
# 1. Bridge Connectivity Sign Dictionary Dataset (WebDataset Shards)
# ============================================================================
class BridgeConnWebDataset(Dataset):
    """
    Streams WebDataset .tar shards or extracts pose-mediapipe pre-computed landmark arrays.
    """
    def __init__(
        self,
        shard_paths: List[str],
        gloss_map: MasterGlossMap,
        target_len: int = 30,
        cache_dir: Optional[str] = None
    ):
        self.shard_paths = shard_paths
        self.gloss_map = gloss_map
        self.target_len = target_len
        self.cache_dir = cache_dir
        self.samples: List[Tuple[str, str]] = []  # List of (data_path_or_key, gloss_label)

        self._index_shards()

    def _index_shards(self):
        """Indexes available shards or mock pre-processed samples."""
        for shard in self.shard_paths:
            if os.path.isdir(shard):
                # Search for pre-computed .npy arrays and .json metadata
                for npy_file in glob.glob(os.path.join(shard, "**/*.npy"), recursive=True):
                    json_file = npy_file.replace(".npy", ".json")
                    gloss = "unknown"
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r') as f:
                                meta = json.load(f)
                                gloss = meta.get("gloss", meta.get("label", "unknown"))
                        except Exception:
                            pass
                    else:
                        gloss = os.path.basename(os.path.dirname(npy_file))
                    self.samples.append((npy_file, gloss))
            elif os.path.isfile(shard) and shard.endswith(".tar"):
                # Handle WebDataset .tar shard reading
                try:
                    import tarfile
                    with tarfile.open(shard, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith(".json"):
                                f = tar.extractfile(member)
                                if f:
                                    meta = json.load(f)
                                    gloss = meta.get("gloss", meta.get("label", "unknown"))
                                    npy_name = member.name.replace(".json", ".npy")
                                    self.samples.append((f"tar://{shard}#{npy_name}", gloss))
                except Exception as e:
                    print(f"[BridgeConnWebDataset] Warning reading tar {shard}: {e}")

        # If no samples found, generate synthetic entries for pipeline testing
        if len(self.samples) == 0:
            print("[BridgeConnWebDataset] No local tar shards found. Creating synthetic fallback samples.")
            dummy_glosses = ["hello", "thank_you", "water", "name", "help"]
            for i in range(50):
                g = dummy_glosses[i % len(dummy_glosses)]
                self.samples.append((f"synthetic_bridgeconn_{i}", g))

        for _, g in self.samples:
            self.gloss_map.get_or_add(g)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path_key, gloss = self.samples[idx]
        label_id = self.gloss_map.get_id(gloss)

        if path_key.startswith("synthetic_"):
            # Generate dummy [30, 258] tensor
            features = np.random.randn(30, 258).astype(np.float32)
        elif os.path.exists(path_key) and path_key.endswith(".npy"):
            raw_data = np.load(path_key)
            features = temporal_resample_or_pad(raw_data, self.target_len)
        else:
            features = np.zeros((self.target_len, 258), dtype=np.float32)

        return torch.from_numpy(features).float(), label_id


# ============================================================================
# 2. INCLUDE Dataset (IIIT Hyderabad - Nested Directories .MOV/.mp4)
# ============================================================================
class IncludeDataset(Dataset):
    """
    Ingests raw .MOV / .mp4 video files organized in nested directory structure:
    root_dir/<Category>/<Gloss>/*.mp4 or root_dir/<Gloss>/*.mp4
    """
    def __init__(
        self,
        root_dir: str,
        gloss_map: MasterGlossMap,
        target_len: int = 30,
        extractor: Optional[MediaPipeFeatureExtractor] = None
    ):
        self.root_dir = root_dir
        self.gloss_map = gloss_map
        self.target_len = target_len
        self.extractor = extractor
        self.samples: List[Tuple[str, str]] = []  # (video_path, gloss_label)

        self._discover_videos()

    def _discover_videos(self):
        if os.path.exists(self.root_dir):
            video_extensions = ("*.mp4", "*.MOV", "*.avi", "*.npy")
            for ext in video_extensions:
                for video_path in glob.glob(os.path.join(self.root_dir, "**", ext), recursive=True):
                    # Directory structure: parent folder name is gloss label
                    gloss = os.path.basename(os.path.dirname(video_path))
                    self.samples.append((video_path, gloss))
                    self.gloss_map.get_or_add(gloss)

        if len(self.samples) == 0:
            print("[IncludeDataset] No raw videos found in root_dir. Creating synthetic fallback samples.")
            dummy_glosses = ["include", "father", "mother", "school", "friend"]
            for i in range(50):
                g = dummy_glosses[i % len(dummy_glosses)]
                self.samples.append((f"synthetic_include_{i}", g))
                self.gloss_map.get_or_add(g)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, gloss = self.samples[idx]
        label_id = self.gloss_map.get_id(gloss)

        if video_path.startswith("synthetic_"):
            features = np.random.randn(self.target_len, 258).astype(np.float32)
        elif video_path.endswith(".npy"):
            raw_data = np.load(video_path)
            features = temporal_resample_or_pad(raw_data, self.target_len)
        elif self.extractor and os.path.exists(video_path):
            raw_features = self.extractor.process_video_path(video_path)
            features = temporal_resample_or_pad(raw_features, self.target_len)
        else:
            features = np.zeros((self.target_len, 258), dtype=np.float32)

        return torch.from_numpy(features).float(), label_id


# ============================================================================
# 3. CISLR Corpus Dataset (Raw .mp4 + dataset.csv mapping)
# ============================================================================
class CislrDataset(Dataset):
    """
    Ingests raw .mp4 clips mapped dynamically to gloss labels via dataset.csv.
    CSV format expected: filename,gloss (or video_name,label).
    """
    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        gloss_map: MasterGlossMap,
        target_len: int = 30,
        extractor: Optional[MediaPipeFeatureExtractor] = None
    ):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.gloss_map = gloss_map
        self.target_len = target_len
        self.extractor = extractor
        self.samples: List[Tuple[str, str]] = []

        self._load_csv()

    def _load_csv(self):
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename", row.get("video_name", row.get("video", "")))
                    gloss = row.get("gloss", row.get("label", row.get("sign", "")))
                    if fname and gloss:
                        vpath = os.path.join(self.root_dir, fname)
                        self.samples.append((vpath, gloss))
                        self.gloss_map.get_or_add(gloss)

        if len(self.samples) == 0:
            print("[CislrDataset] CSV or video directory not found. Creating synthetic fallback samples.")
            dummy_glosses = ["cislr_sign_1", "cislr_sign_2", "water", "hello", "book"]
            for i in range(50):
                g = dummy_glosses[i % len(dummy_glosses)]
                self.samples.append((f"synthetic_cislr_{i}", g))
                self.gloss_map.get_or_add(g)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, gloss = self.samples[idx]
        label_id = self.gloss_map.get_id(gloss)

        if video_path.startswith("synthetic_"):
            features = np.random.randn(self.target_len, 258).astype(np.float32)
        elif video_path.endswith(".npy"):
            raw_data = np.load(video_path)
            features = temporal_resample_or_pad(raw_data, self.target_len)
        elif self.extractor and os.path.exists(video_path):
            raw_features = self.extractor.process_video_path(video_path)
            features = temporal_resample_or_pad(raw_features, self.target_len)
        else:
            features = np.zeros((self.target_len, 258), dtype=np.float32)

        return torch.from_numpy(features).float(), label_id


# ============================================================================
# 4 & 5. Mendeley Data & Kaggle Datasets (Custom JSON Map Alignment)
# ============================================================================
class MendeleyKaggleDataset(Dataset):
    """
    Ingests local .mp4 clips from Mendeley & Kaggle (harsh0239 & arvindvinod).
    Uses a custom JSON mapping dictionary to align diverse folder/file naming
    conventions into the MasterGlossMap.
    """
    def __init__(
        self,
        root_dir: str,
        custom_json_map_path: Optional[str],
        gloss_map: MasterGlossMap,
        target_len: int = 30,
        extractor: Optional[MediaPipeFeatureExtractor] = None
    ):
        self.root_dir = root_dir
        self.custom_map: Dict[str, str] = {}
        self.gloss_map = gloss_map
        self.target_len = target_len
        self.extractor = extractor
        self.samples: List[Tuple[str, str]] = []

        if custom_json_map_path and os.path.exists(custom_json_map_path):
            with open(custom_json_map_path, 'r', encoding='utf-8') as f:
                self.custom_map = json.load(f)

        self._discover_clips()

    def _discover_clips(self):
        if os.path.exists(self.root_dir):
            for ext in ("*.mp4", "*.avi", "*.npy"):
                for video_path in glob.glob(os.path.join(self.root_dir, "**", ext), recursive=True):
                    raw_dir_name = os.path.basename(os.path.dirname(video_path))
                    # Align folder name via custom map if available
                    master_gloss = self.custom_map.get(raw_dir_name, raw_dir_name)
                    self.samples.append((video_path, master_gloss))
                    self.gloss_map.get_or_add(master_gloss)

        if len(self.samples) == 0:
            print("[MendeleyKaggleDataset] Directory not found. Creating synthetic fallback samples.")
            dummy_glosses = ["hello", "thank_you", "please", "yes", "no"]
            for i in range(50):
                g = dummy_glosses[i % len(dummy_glosses)]
                self.samples.append((f"synthetic_mendeley_{i}", g))
                self.gloss_map.get_or_add(g)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, gloss = self.samples[idx]
        label_id = self.gloss_map.get_id(gloss)

        if video_path.startswith("synthetic_"):
            features = np.random.randn(self.target_len, 258).astype(np.float32)
        elif video_path.endswith(".npy"):
            raw_data = np.load(video_path)
            features = temporal_resample_or_pad(raw_data, self.target_len)
        elif self.extractor and os.path.exists(video_path):
            raw_features = self.extractor.process_video_path(video_path)
            features = temporal_resample_or_pad(raw_features, self.target_len)
        else:
            features = np.zeros((self.target_len, 258), dtype=np.float32)

        return torch.from_numpy(features).float(), label_id


# ============================================================================
# 6. Federated ISL ConcatDataset Wrapper
# ============================================================================
class FederatedISLDataset:
    """
    Federated Dataset Builder.
    Unifies all 5 ISL dataset sources into a single PyTorch ConcatDataset.
    """
    def __init__(
        self,
        bridgeconn_shards: List[str] = [],
        include_dir: str = "",
        cislr_dir: str = "",
        cislr_csv: str = "",
        mendeley_dir: str = "",
        kaggle_dir: str = "",
        custom_json_map: Optional[str] = None,
        master_gloss_map: Optional[MasterGlossMap] = None,
        target_len: int = 30,
        extractor: Optional[MediaPipeFeatureExtractor] = None
    ):
        self.gloss_map = master_gloss_map if master_gloss_map is not None else MasterGlossMap()
        self.datasets: List[Dataset] = []

        # 1. Bridge Connectivity Dataset
        ds1 = BridgeConnWebDataset(bridgeconn_shards, self.gloss_map, target_len)
        self.datasets.append(ds1)

        # 2. INCLUDE Dataset
        ds2 = IncludeDataset(include_dir, self.gloss_map, target_len, extractor)
        self.datasets.append(ds2)

        # 3. CISLR Corpus Dataset
        ds3 = CislrDataset(cislr_dir, cislr_csv, self.gloss_map, target_len, extractor)
        self.datasets.append(ds3)

        # 4. Mendeley Dataset
        ds4 = MendeleyKaggleDataset(mendeley_dir, custom_json_map, self.gloss_map, target_len, extractor)
        self.datasets.append(ds4)

        # 5. Kaggle Dataset
        ds5 = MendeleyKaggleDataset(kaggle_dir, custom_json_map, self.gloss_map, target_len, extractor)
        self.datasets.append(ds5)

        # Combine into unified ConcatDataset
        self.concat_dataset = ConcatDataset(self.datasets)

    def get_dataset(self) -> ConcatDataset:
        return self.concat_dataset

    def get_gloss_map(self) -> MasterGlossMap:
        return self.gloss_map

    def summary(self):
        print("=== Federated ISL Dataset Summary ===")
        print(f"Total Combined Samples: {len(self.concat_dataset)}")
        print(f"Total Unique Vocabulary Classes: {len(self.gloss_map)}")
        for idx, ds in enumerate(self.datasets, 1):
            print(f"  Source {idx} ({ds.__class__.__name__}): {len(ds)} samples")
        print("=====================================")


def create_federated_dataloaders(
    federated_ds: FederatedISLDataset,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, MasterGlossMap]:
    """
    Splits the unified ConcatDataset into train and validation DataLoaders.
    """
    dataset = federated_ds.get_dataset()
    gloss_map = federated_ds.get_gloss_map()

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, gloss_map


if __name__ == "__main__":
    # Test federated dataset construction
    fed_ds = FederatedISLDataset()
    fed_ds.summary()
    train_loader, val_loader, gmap = create_federated_dataloaders(fed_ds, batch_size=8)
    
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}, Batch Y shape: {batch_y.shape}")
        break
