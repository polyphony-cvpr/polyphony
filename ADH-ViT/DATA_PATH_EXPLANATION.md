# Understanding `data_path` vs `data_root` in ADH-ViT

## TL;DR

**In 99% of cases, you only need to specify `--lh_data_path` and `--rh_data_path`!**

The `--lh_data_root` and `--rh_data_root` arguments are **optional** and will automatically default to the corresponding `data_path` values.

---

## Quick Example

### ✅ Simplified (Recommended)
```bash
python main.py \
    --lh_data_path /path/to/havid/lh_v0 \
    --rh_data_path /path/to/havid/rh_v0 \
    [... other args ...]
```

### ✅ Advanced (Only if needed)
```bash
python main.py \
    --lh_data_path /path/to/annotations/lh \
    --lh_data_root /different/path/to/videos/lh \
    --rh_data_path /path/to/annotations/rh \
    --rh_data_root /different/path/to/videos/rh \
    [... other args ...]
```

---

## What's the Difference?

### `--lh_data_path` / `--rh_data_path`

**Purpose:** Path to the directory containing annotation files

**Used for:**
- Finding `train_list_video.txt`
- Finding `val_list_video.txt`

**Example:**
```python
anno_path = os.path.join(args.data_path, 'train_list_video.txt')
```

### `--lh_data_root` / `--rh_data_root`

**Purpose:** Root directory where video files are stored

**Used for:**
- Passed to the dataset constructor
- Used to locate actual video files listed in the annotation files

**Example:**
```python
dataset = VideoClsDataset(
    anno_path=anno_path,
    data_root=args.data_root,  # Videos are found here
    ...
)
```

---

## Typical Dataset Structure (99% of cases)

```
/path/to/havid/lh_v0/                  # This is your lh_data_path
├── train_list_video.txt               # Annotation file
├── val_list_video.txt                 # Annotation file
└── videos/                            # Video files
    ├── action1/
    │   └── video1.mp4
    ├── action2/
    │   └── video2.mp4
    └── ...
```

In this case:
- `--lh_data_path /path/to/havid/lh_v0`
- `--lh_data_root` will automatically default to `/path/to/havid/lh_v0`

**✅ You only need to specify `data_path`!**

---

## When You Need Different Paths (Rare)

Sometimes annotations and videos are stored separately:

```
/path/to/annotations/lh/               # This is your lh_data_path
├── train_list_video.txt
└── val_list_video.txt

/different/storage/videos/lh/          # This is your lh_data_root
└── videos/
    ├── action1/
    │   └── video1.mp4
    └── ...
```

In this case, you need BOTH:
- `--lh_data_path /path/to/annotations/lh`
- `--lh_data_root /different/storage/videos/lh`

**This is useful for:**
- Annotations on fast SSD, videos on slow HDD
- Shared annotation files with different video locations
- Network storage scenarios

---

## Examples

### Example 1: Standard Dual-Hand Training (Simplified)

```bash
python main.py \
    --model vit_base_patch16_224_alternating \
    --lh_data_path /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --rh_data_path /home/hao/Polyphony/data/havid_mmaction/rh_v0 \
    --lh_num_classes 75 \
    --rh_num_classes 75 \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/havid \
    --batch_size 4 \
    --epochs 50
```

### Example 2: Single-Stream Training (Even Simpler!)

```bash
python main.py \
    --one_stream \
    --lh_data_path /home/hao/Polyphony/data/single_stream \
    --lh_num_classes 75 \
    --finetune models/vit_b_k710_dl_from_giant.pth \
    --output_dir output/single_stream \
    --batch_size 4 \
    --epochs 50
```

Note: `--rh_data_path` is not needed with `--one_stream`!

### Example 3: Separate Annotation and Video Storage

```bash
python main.py \
    --lh_data_path /fast_ssd/annotations/lh \
    --lh_data_root /slow_hdd/videos/lh \
    --rh_data_path /fast_ssd/annotations/rh \
    --rh_data_root /slow_hdd/videos/rh \
    [... other args ...]
```

---

## How the Code Works

```python
# In main.py, if data_root is not specified, it defaults to data_path:
if not args.lh_data_root:
    args.lh_data_root = args.lh_data_path
if not args.rh_data_root and args.rh_data_path:
    args.rh_data_root = args.rh_data_path

# Then both are used:
# 1. data_path: to find annotation files
anno_path = os.path.join(args.data_path, 'train_list_video.txt')

# 2. data_root: to load actual videos
dataset = VideoClsDataset(
    anno_path=anno_path,
    data_root=args.data_root,  # Where videos are located
    ...
)
```

---

## Annotation File Format

The annotation files (`train_list_video.txt`, `val_list_video.txt`) contain:

```
action1/video1.mp4 0
action1/video2.mp4 0
action2/video3.mp4 1
action2/video4.mp4 1
...
```

The video paths in these files are **relative to `data_root`**, not `data_path`!

So if `data_root=/path/to/videos` and the annotation says `action1/video1.mp4`, the full path becomes:
```
/path/to/videos/action1/video1.mp4
```

---

## Common Mistakes

### ❌ Mistake 1: Specifying both when not needed

```bash
# Redundant! data_root will default to data_path anyway
python main.py \
    --lh_data_path /path/to/havid/lh \
    --lh_data_root /path/to/havid/lh \  # Not needed!
    --rh_data_path /path/to/havid/rh \
    --rh_data_root /path/to/havid/rh \  # Not needed!
    [...]
```

### ✅ Fix: Remove redundant arguments

```bash
python main.py \
    --lh_data_path /path/to/havid/lh \
    --rh_data_path /path/to/havid/rh \
    [...]
```

### ❌ Mistake 2: Wrong data_root when they should be different

```bash
# If annotations and videos are in different places
python main.py \
    --lh_data_path /annotations/lh \
    # Missing: --lh_data_root /videos/lh
    [...]
# This will fail because it looks for videos in /annotations/lh/
```

### ✅ Fix: Specify both paths

```bash
python main.py \
    --lh_data_path /annotations/lh \
    --lh_data_root /videos/lh \
    [...]
```

---

## Summary Table

| Scenario | Specify `data_path` | Specify `data_root` |
|----------|---------------------|---------------------|
| Standard structure (annotations + videos together) | ✅ Yes | ❌ No (defaults to `data_path`) |
| Separate annotations and videos | ✅ Yes | ✅ Yes |
| Single-stream training | ✅ Yes (lh only) | ❌ No |

---

## Updated Training Script

The `train.sh` script has been simplified:

**Before:**
```bash
LH_DATA_PATH='../data/havid/left_hand'
LH_DATA_ROOT='../data/havid/left_hand'  # Redundant!
RH_DATA_PATH='../data/havid/right_hand'
RH_DATA_ROOT='../data/havid/right_hand'  # Redundant!

python main.py \
    --lh_data_path ${LH_DATA_PATH} \
    --lh_data_root ${LH_DATA_ROOT} \
    --rh_data_path ${RH_DATA_PATH} \
    --rh_data_root ${RH_DATA_ROOT} \
    [...]
```

**After:**
```bash
LH_DATA_PATH='../data/havid/left_hand'
RH_DATA_PATH='../data/havid/right_hand'

python main.py \
    --lh_data_path ${LH_DATA_PATH} \
    --rh_data_path ${RH_DATA_PATH} \
    [...]
```

**Much cleaner!** 🎉

---

## Key Takeaways

1. **`data_path`**: Where your annotation files are (`.txt` files)
2. **`data_root`**: Where your video files are (`.mp4` files)
3. **In most cases**: They're the same, so only specify `data_path`
4. **`data_root` automatically defaults** to `data_path` if not specified
5. **Only specify `data_root`** if videos are in a different location

---

## Questions?

- Check `README.md` for full documentation
- Check `HOWTO_RUN.md` for training examples
- Check `SINGLE_STREAM_TRAINING.md` for `--one_stream` mode

This simplification makes ADH-ViT easier to use while maintaining flexibility for advanced scenarios!

