import torch
import time
import math

def run_calibrated_swa(max_iters=1000):
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type=device)
    
    swa_weights = None
    swa_count = 0
    swa_start = 800
    
    print(f"Kalibre Edilmiş SWA Deneyi Başlıyor...")
    start_time = time.time()

    for iter in range(1, max_iters + 1):
        # One-Cycle LR Schedule
        decay_ratio = max(0, (iter - 300) / 700) if iter > 300 else 0
        lr = 1e-3 * (iter/300) if iter <= 300 else 1e-3 * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        X, Y = get_batch()
        logits, loss = model(X, Y)
        
        # Calibration: Gradyan adımından önce logitleri hafifçe ölçeklendir (Temperature = 0.9)
        # Bu, modelin eğitim sırasında daha 'keskin' kararlar vermesini sağlar
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iter >= swa_start:
            with torch.no_grad():
                if swa_weights is None:
                    swa_weights = [p.clone().detach() for p in model.parameters()]
                else:
                    for i, p in enumerate(model.parameters()):
                        swa_weights[i] = (swa_weights[i] * swa_count + p.detach()) / (swa_count + 1)
            swa_count += 1

    # SWA Finalize
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(swa_weights[i])
        
        X_test, Y_test = get_batch()
        logits, final_loss_obj = model(X_test, Y_test)
        final_loss = final_loss_obj.item()
        
        # Doğruluk hesaplama (Calibrated Top-1)
        probs = torch.softmax(logits, dim=-1)
        acc = (probs.argmax(dim=-1) == Y_test).float().mean().item() * 100

    duration = time.time() - start_time
    print(f"\n--- KALİBRE EDİLMİŞ SONUÇ ---")
    print(f"Final Loss: {final_loss:.4f} | Süre: {duration:.2f}s | Doğruluk: %{acc:.2f}")
    return final_loss, acc, duration

loss_calib, acc_calib, dur_calib = run_calibrated_swa(1000)
