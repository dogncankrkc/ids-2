import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.
    
    Amaç: 
    Standart Cross Entropy, kolay sınıflandırılan örneklerin baskınlığında kalır. 
    Focal Loss, modelin "zaten bildiği" örneklerin loss değerini (gamma ile) 
    kısıp, enerjisini "zor/azınlık" sınıflara (Web Attack, Backdoor vb.) harcamasını sağlar.
    
    Parameters:
        alpha (Tensor, optional): Sınıf ağırlıkları (Class balancing için).
        gamma (float): Odaklanma parametresi. (2.0 veya 2.5 genelde idealdir).
        reduction (str): 'mean', 'sum' veya 'none'.
        device (str/torch.device): GPU veya CPU.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        # Alpha (Sınıf Ağırlıkları) kontrolü
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] (Modelin çıktısı / Logits) - Softmax uygulanmamış ham değerler
            targets: [N] (Gerçek sınıf indeksleri)
        """
        
        # 1. Standart Cross Entropy Loss hesapla (Reduction olmadan, ham loss lazım)
        # weight=self.alpha diyerek dengesiz veri setini yönetiyoruz.
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # 2. Modelin o sınıfı tahmin etme olasılığını (pt) bul
        # Formül: log(pt) = -ce_loss  =>  pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Focal Loss Formülü: (1 - pt)^gamma * CE_Loss
        # pt yüksekse (kolay örnek), (1-pt) sıfıra yaklaşır -> Loss düşer.
        # pt düşükse (zor örnek), (1-pt) bire yaklaşır -> Loss olduğu gibi kalır (cezalandırır).
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Reduction (Ortalama veya Toplam)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss