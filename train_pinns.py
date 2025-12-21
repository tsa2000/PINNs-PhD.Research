import torch
import torch.nn as nn
import numpy as np
import copy

# 1. إعداد الثوابت والفيزياء
torch.manual_seed(42)
np.random.seed(42)

M, CP, A = 0.042, 800.0, 0.004185
T_INF, H_BASELINE = 23.0, 20.0
SIGMA, EPSILON = 5.67e-8, 0.85
TIME_SCALE, T_SCALE, V_SCALE = 60.0, 1000.0, 200.0

def get_q_tr(t):
    q_peak = 11040.0
    val = torch.zeros_like(t)
    mask1 = (t < 1.0) & (t >= 0.0)
    val[mask1] = q_peak * t[mask1]
    mask2 = (t >= 1.0) & (t <= 5.0)
    val[mask2] = q_peak * (1.0 - (t[mask2] - 1.0) / 4.0)
    return val

def get_h(v_kmh):
    v_ms = v_kmh / 3.6
    h_forced = 5.0 + 4.0 * (v_ms ** 0.8) 
    return torch.clamp(h_forced, min=H_BASELINE)

# 2. بناء الشبكة العصبية (Heavy-Duty Architecture)
class BatteryPINN_Uncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, t, v):
        inputs = torch.cat([t/TIME_SCALE, v/V_SCALE], dim=1)
        return self.net(inputs) * T_SCALE

# 3. دالة الخسارة الفيزيائية
def physics_loss(model, t, v):
    t.requires_grad = True
    T_pred = model(t, v)
    dTdt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    
    Q_gen, h_val = get_q_tr(t), get_h(v)
    Q_conv = h_val * A * (T_pred - T_INF)
    Q_rad = EPSILON * SIGMA * A * ((T_pred+273.15)**4 - (T_INF+273.15)**4)
    
    residual = (M * CP * dTdt) - (Q_gen - Q_conv - Q_rad)
    return torch.mean((residual / (M*CP))**2)

# 4. حلقة التدريب الرئيسية
if __name__ == "__main__":
    model = BatteryPINN_Uncertainty()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)

    print("🚀 Starting Training...")
    for epoch in range(8001):
        optimizer.zero_grad()
        
        # نقاط عشوائية للتدريب
        t_phy = torch.rand(2000, 1) * 60.0
        v_phy = torch.rand(2000, 1) * 200.0
        loss_phy = physics_loss(model, t_phy, v_phy)
        
        # الشروط الابتدائية
        t_ic = torch.zeros(500, 1)
        v_ic = torch.rand(500, 1) * 200.0
        loss_ic = torch.mean((model(t_ic, v_ic) - T_INF)**2)
        
        total_loss = loss_phy + 100.0 * loss_ic
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.5f}")

    torch.save(model.state_dict(), "battery_pinn_uncertainty.pth")
    print("✅ Training Complete. Model saved as 'battery_pinn_uncertainty.pth'")
