"""
GRU Architecture for IDS (Stability Focused)
Target: Eliminate Validation Fluctuations
Mechanism: Uses LayerNorm instead of BatchNorm to handle SMOTE/Real data mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IDS_GRU(nn.Module):
    def __init__(self, num_classes=8, input_dim=39, hidden_dim=128, num_layers=2):
        super(IDS_GRU, self).__init__()
        
        # GRU, veriyi (Batch, Seq_Len, Features) formatında ister.
        # Biz 39 özelliği "39 adımlık bir zaman serisi" gibi vereceğiz.
        # Input Size: 1 (Her adımda 1 özellik giriyor)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Feature Projection (Özellikleri biraz zenginleştir)
        # 1 -> 16'ya çıkarıyoruz ki GRU daha rahat öğrensin
        self.embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        # 2. GRU Gövdesi
        # Dropout: Katmanlar arası unutma
        self.gru = nn.GRU(
            input_size=16, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3
        )
        
        # 3. Stabilizasyon (DALGALANMAYI BİTİRECEK KISIM)
        # LayerNorm, Batch istatistiklerine bakmaz, her örneği kendi içinde normalize eder.
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Sınıflandırıcı
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 39) veya (Batch, 1, 39)
        
        # GRU için boyut düzeltme: (Batch, 39, 1) yapmalıyız
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (N, 39) -> (N, 39, 1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.permute(0, 2, 1) # (N, 1, 39) -> (N, 39, 1)

        # Embedding: (N, 39, 1) -> (N, 39, 16)
        x = self.embedding(x)

        # GRU: Çıktı (Batch, Seq_Len, Hidden)
        # h_n (Hidden State) kullanacağız: (Num_Layers, Batch, Hidden)
        out, _ = self.gru(x)
        
        # Son zaman adımının çıktısını al (Many-to-One)
        # out[:, -1, :] -> Tüm batch, Son adım, Tüm hidden featurelar
        final_feature = out[:, -1, :]
        
        # LayerNorm uygula (Batch Norm yerine)
        final_feature = self.layer_norm(final_feature)
        
        # Sınıflandır
        logits = self.fc(final_feature)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_ids_model(mode: str = "multiclass", num_classes: int = 8, input_dim: int = 39):
    print(f"[FACTORY] Initializing IDS_GRU (Stability Edition).")
    # Raspberry Pi için optimize edilmiş parametreler
    model = IDS_GRU(num_classes=num_classes, input_dim=input_dim, hidden_dim=128, num_layers=2)
    return model