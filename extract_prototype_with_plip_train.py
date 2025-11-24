import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPTextModel
import json

target_dim = 256
batch_size = 8
num_epochs = 100
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextFeatureDataset(Dataset):
    def __init__(self, cls_features: torch.Tensor):
        self.cls_features = cls_features

    def __len__(self):
        return self.cls_features.size(0)

    def __getitem__(self, idx):
        return self.cls_features[idx]

class TextAutoEncoder(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=256):
        super().__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out

def get_path_prorotypes():

    # load plip
    processor = CLIPProcessor.from_pretrained("vinid/plip")
    text_encoder = CLIPTextModel.from_pretrained("vinid/plip")
    text_encoder.eval().to(device)

    # train the projector

    with open("prompt.txt", "r") as f:
        prompt_dict = json.load(f)

    all_prompts = []
    for cat, plist in prompt_dict.items():
        all_prompts.extend(plist)

    cls_embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = text_encoder(**inputs)
            cls_batch = outputs.last_hidden_state[:, 0, :]  # [B, 768]
            cls_embeddings.append(cls_batch.cpu())

    cls_embeddings = torch.cat(cls_embeddings, dim=0)  # [N, 768]
    dataset = TextFeatureDataset(cls_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化 AutoEncoder
    ae = TextAutoEncoder(input_dim=512, bottleneck_dim=256).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    ae.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            _, reconstructed = ae(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1:03d}/{num_epochs}, MSE: {avg_loss:.6f}")

    projector = ae.encoder

    categories = list(prompt_dict.keys())
    n_proto = len(categories)


    # initialize prototype tensor and encoder token
    p_proto = torch.zeros((1, n_proto, target_dim), dtype=torch.float32).to(device)

    for idx, cat in enumerate(categories):
        prompts = prompt_dict[cat]
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim]
            embeds_256 = projector(cls_embeddings)  # [B, 256]
            avg_embed = torch.mean(embeds_256, dim=0, keepdim=True)  # [1, 256]
            p_proto[0, idx, :] = avg_embed

    # save
    # torch.save({"p_proto": p_proto.cpu(), "p_encoder_token": p_encoder_token.cpu(), "categories": categories}, "plip_prototypes.pt")

    print("Path prototypes extracted")
    
    del text_encoder
    del processor
    del projector
    torch.cuda.empty_cache()
    return p_proto
