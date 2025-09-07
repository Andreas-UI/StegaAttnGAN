import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import tqdm
import torchvision
from torchvision import transforms

from StegaAttnGAN import training_step, StegaConfig, StegaAttnGAN, rand_tokens_with_pad, to_minus1_1

device = "cuda" if torch.cuda.is_available() else "cpu"

max_msg_len=3
cfg = StegaConfig()
cfg.max_msg_len = 2
cfg.vocab_size = 128

batch_size = 512 * 2

model = StegaAttnGAN(cfg).to(device)

# # --- Optimizers ---
opt_g = torch.optim.AdamW(list(model.msg_enc.parameters())+list(model.gen.parameters())+list(model.dec.parameters()), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.AdamW(model.disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

# # --- Random message generator (fixed length = cfg.max_msg_len) ---
def rand_tokens(B: int, L: int, vocab: int):
    return torch.randint(0, vocab, (B, L), dtype=torch.long)

# os.makedirs('samples', exist_ok=True)
global_step = 0

# min_msg_len = 1
# max_msg_len = cfg.max_msg_len  # 5
# base_epochs = 10  # for length = 1
# increment = 2   # extra epochs per length increase

min_msg_len = 1
max_msg_len = cfg.max_msg_len
base_epochs = 200  # for length = 1
increment = 100    # extra epochs per length increase


schedule = []
total_epochs = 0
for length in range(min_msg_len, max_msg_len + 1):
    total_for_length = base_epochs + (length - min_msg_len) * increment
    # Split evenly between text and image
    text_epochs = total_for_length // 2
    image_epochs = total_for_length - text_epochs  # handles odd numbers if needed
    schedule.append((length, "text", text_epochs))
    schedule.append((length, "image", image_epochs))
    total_epochs += total_for_length

# Expand schedule into per-epoch mapping
epoch_nums = []
msg_lengths = []
phases = []
phase_count = {}

epoch_counter = 1
for length, phase, num_epochs in schedule:
    for _ in range(num_epochs):
        epoch_nums.append(epoch_counter)
        msg_lengths.append(length)
        phases.append(phase)
        phase_count[phase] = phase_count.get(phase, 0) + 1
        epoch_counter += 1

print(phase_count)

# ---- Training loop ----
epoch_counter = 0

# ---- Fake Dataset ----
class RandomImageDataset(Dataset):
    def __init__(self, num_samples=40, image_size=(3, 64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random float tensor in [0,1]
        return torch.rand(self.image_size)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(to_minus1_1),
])

cifar10_train = torchvision.datasets.CIFAR10(
    root='./data',
    download=True,
    transform=transform,
    train=True
)

class ImageOnlyWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image  # discard label

trainloader = torch.utils.data.DataLoader(
    ImageOnlyWrapper(ConcatDataset([cifar10_train])),
    # ImageOnlyWrapper(ConcatDataset([cifar10_train, cifar100_train, food101_train, celeba_train, dtd_train])),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=torch.get_num_threads()-10,
    shuffle=True,
    prefetch_factor=8,
    persistent_workers=True
)

# ---- DataLoader ----
train_dataset = RandomImageDataset(num_samples=100, image_size=(3, 32, 32))
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# ---- Check the loop ----
for batch in trainloader:
    print(batch.shape)
    break


def train_one_epoch(length, phase, epoch_idx):
    pbar = tqdm.tqdm(trainloader, desc=f"Epoch {epoch_idx+1}/{total_epochs} [{phase}:{length}]", unit="batch")
    global global_step
    for images in pbar:
        images = images.to(device, non_blocking=True)
        B = images.size(0)
        tokens, token_pad_mask = rand_tokens_with_pad(B, length, cfg.max_msg_len, cfg.vocab_size)
        tokens = tokens.to(device)
        token_pad_mask = token_pad_mask.to(device)
        batch = (images, tokens, token_pad_mask, cfg.max_msg_len)
        log = training_step(model, batch, opt_g, opt_d, cfg, phase=phase)
        global_step += 1
        pbar.set_postfix({
            'PSNR': f"{log['psnr']:.2f}",
            'img': f"{log['img_loss']:.4f}",
            'txt': f"{log['txt_loss']:.4f}",
            # 'g_gan': f"{log['gan_g']:.4f}",
            # 'd_gan': f"{log['gan_d']:.4f}"
            'gen_total': f"{log['gen_total']:.4f}"
        })

    return log['gen_total']

for length, phase, num_epochs in schedule:
    for _ in range(num_epochs):
        loss = train_one_epoch(length, phase, epoch_counter)
        epoch_counter += 1


print("Training complete.")
