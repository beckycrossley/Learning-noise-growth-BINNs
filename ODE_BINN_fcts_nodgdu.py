import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
import os
import time
import math
import json
import torch
import random
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from scipy import stats
from scipy.optimize import curve_fit
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class fcts:
    def load_dataset_npz(npz_path, n_grid=400):
        d = np.load(npz_path, allow_pickle=True)
        t = d["t"].astype(np.float32).ravel()
        C_true = d["C_true"].astype(np.float32).ravel()
        C_reps = d["C_reps"].astype(np.float32)   # (R, T)

        # construct a consistent C_grid for parametric evaluation
        Cmin = float(np.min(C_true))
        Cmax = float(np.max(C_true))
        C_grid = np.linspace(Cmin, Cmax, n_grid, dtype=np.float32)

        return {
            "t": t,
            "C_true": C_true,
            "C_reps": C_reps,
            "C_list": [C_reps[r].ravel() for r in range(C_reps.shape[0])],
            "C_grid": C_grid
        }

    def _rk4_step_scalar(f, u, t, dt):
        k1 = f(u, t)
        k2 = f(u + 0.5*dt*k1, t + 0.5*dt)
        k3 = f(u + 0.5*dt*k2, t + 0.5*dt)
        k4 = f(u + dt*k3, t + dt)
        return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # ---------- Simulate ODE forward using g_net ----------
    def simulate_forward_from_gnet(g_net, t_grid, C0, device="cpu", max_substeps=10):
        g_net = g_net.to(device)
        g_net.eval()
        t_grid = np.asarray(t_grid).ravel()
        nT = len(t_grid)
        u_out = np.empty(nT, dtype=np.float32)
        u = float(C0)
        u_out[0] = u

        # helper to evaluate du/dt = u*g(u) using g_net
        def f(u_scalar, t_scalar):
            # convert to tiny tensor and evaluate g_net
            with torch.no_grad():
                ut = torch.as_tensor([[u_scalar]], dtype=torch.float32, device=device)
                gval = g_net(ut)         # tensor shape (1,1)
                gv = float(gval.squeeze().cpu().numpy())
            return u_scalar * gv

        for i in range(nT - 1):
            t0 = float(t_grid[i]); t1 = float(t_grid[i+1])
            dt_full = t1 - t0
            # choose number of substeps proportional to dt (and bounded)
            # ensures numerical stability when dt large; you can tune max_substeps
            nsub = max(1, min(max_substeps, int(np.ceil(dt_full / ( (t_grid[1]-t_grid[0]) if nT>1 else dt_full )))))
            dt = dt_full / nsub
            for s in range(nsub):
                u = fcts._rk4_step_scalar(f, u, t0 + s*dt, dt)
                # ensure non-negativity if desired (population)
                if np.isnan(u) or np.isinf(u):
                    # numerical blowup: break and return nans
                    u = np.nan
                    break
            u_out[i+1] = u
        return u_out            

    def compute_mechanistic_forward_rmse(
        save_dir,               # experiment folder containing rep_{01..Numrep}/binn_ode_g_best.pt
        ens_truth,              # ensemble truth dict (must have "t_full" and "_true": {"t","C"} )
        Numrep=10,
        device="cpu",
        use_ckpt_name="binn_ode_g_best.pt",
        initial_condition="ensemble_mean"  # "ensemble_mean" or "true" or "first_rep"
    ):
        """
        For each rep, loads rep_{rep:02d}/binn_ode_g_best.pt to obtain g_net state,
        instantiates a new GNet (must match your GNet constructor signature), loads state_dict,
        simulates ODE forward from chosen initial condition and computes RMSE vs true trajectory.
        Returns:
        - forward_runs: np.array shape (Numrep, T)  (NaN row for missing reps)
        - forward_rmse: np.array shape (Numrep,) RMSE values (np.nan for missing)
        - forward_mean: np.array shape (T,) mean curve across available reps (nanmean)
        """
        t_grid = np.asarray(ens_truth["t_full"]).ravel()
        u_true = np.asarray(ens_truth["_true"]["C"]).ravel()
        T = len(t_grid)

        forward_runs = np.full((Numrep, T), np.nan, dtype=np.float32)
        forward_rmse = np.full((Numrep,), np.nan, dtype=np.float32)

        # choose initial value
        if initial_condition == "true":
            C0 = float(u_true[0])
        elif initial_condition == "ensemble_mean" and "C_full_mean" in ens_truth:
            C0 = float(np.asarray(ens_truth["C_full_mean"]).ravel()[0])
        elif initial_condition == "first_rep" and "C_reps" in ens_truth:
            C0 = float(np.asarray(ens_truth["C_reps"])[0,0])
        else:
            # fallback to true
            C0 = float(u_true[0])

        for rep in range(1, Numrep+1):
            ckpt = os.path.join(save_dir, f"rep_{rep:02d}", use_ckpt_name)
            if not os.path.exists(ckpt):
                # try alternative noise_best.pt -> but that contains only noise model
                print(f"[compute_mechanistic_forward_rmse] missing {ckpt}; skipping rep {rep}")
                continue
            try:
                ck = torch.load(ckpt, map_location=device)
                # instantiate GNet with same hyperparams as you used for training
                # IMPORTANT: match the architecture used in training (num_layers, hidden)
                # If you used variables in scope, replace these with the correct numbers.
                g_net = GNet(num_layers=3, hidden_units=64, activation_fn=torch.nn.Tanh(), input_dim=1, output_dim=1).to(device)
                # ck may hold keys; earlier training saved {"u_net":..., "g_net": ...}
                if "g_net" in ck:
                    g_net.load_state_dict(ck["g_net"])
                else:
                    # maybe entire model saved directly
                    try:
                        g_net.load_state_dict(ck)
                    except Exception as e:
                        print(f"Warning loading g_net state for rep {rep}: {e}")
                        continue

                # simulate forward
                u_forw = fcts.simulate_forward_from_gnet(g_net, t_grid, C0, device=device, max_substeps=12)
                forward_runs[rep-1, :] = u_forw.astype(np.float32)

                # compute rmse but ignore NaNs
                mask = ~np.isnan(u_forw)
                if mask.sum() == 0:
                    forward_rmse[rep-1] = np.nan
                else:
                    forward_rmse[rep-1] = math.sqrt(np.mean((u_forw[mask] - u_true[mask])**2))
            except Exception as e:
                print(f"Error evaluating rep {rep} at {ckpt}: {e}")
                continue

        forward_mean = np.nanmean(forward_runs, axis=0)
        return forward_runs, forward_rmse, forward_mean
    
    def C_log_true(t, r, K, C0):
        return BINNs.logistic_analytic(t, r, K, C0).astype(np.float32)
    
    def C_gom_true(t, r, K, C0):
        return BINNs.gompertz_analytic(t, r, K, C0).astype(np.float32)
    
    def C_rich_true(t, r, K, C0, beta):
        return BINNs.richards_analytic(t, r, K, C0, beta).astype(np.float32)

    # =========================
    # Autograd helpers
    # =========================

    def compute_dC_dt(C, t_input):
        """
        dC/dt via autograd. t_input must require grad.
        """
        dC_dt = torch.autograd.grad(
            outputs=C,
            inputs=t_input,
            grad_outputs=torch.ones_like(C),
            create_graph=True, retain_graph=True
        )[0]
        return dC_dt
    

    def logistic_g(C, r, K):
        return r * (1.0 - C / K)

    def logistic_analytic(t, r, K, C0):
        # C(t) = K C0 / (C0 + (K - C0) e^{-rt})
        return (K * C0) / (C0 + (K - C0) * np.exp(-r * t))

    def gompertz_g(C, r, K):
        return r * np.log(K / C)

    def gompertz_analytic(t, r, K, C0):
        # C(t) = K * exp( log(C0/K) * e^{-rt} )
        return K * np.exp(np.log(C0 / K) * np.exp(-r * t))

    def richards_g(C, r, K, beta):
        return r * (1.0 - (C / K) ** beta)

    def richards_analytic(t, r, K, C0, beta):
        # C(t) = (K C0) / [ C0^β + (K^β - C0^β) e^{-β r t} ]^{1/β}
        denom = (C0**beta + (K**beta - C0**beta) * np.exp(-beta * r * t)) ** (1.0 / beta)
        return (K * C0) / denom

class BINNs:
    # =========================
    # Loss terms
    # =========================

    def ode_loss_multiplicative(t_min, t_max, device, u_net, g_net, N_PDE):
        t = torch.rand(N_PDE, 1, requires_grad=True, device=device)
        t = t * (t_max - t_min) + t_min

        C_pred = u_net(t)
        dC_dt  = fcts.compute_dC_dt(C_pred, t)
        g_pred = g_net(C_pred)

        residual = dC_dt - C_pred * g_pred
        return torch.mean(residual**2)

    def initial_condition_loss(u_net, t0, C0):
        C_pred0 = u_net(t0)
        return torch.mean((C_pred0 - C0)**2)

    def bio_loss(u_net, t_min, t_max):
        N_bio = 256
        t = torch.rand(N_bio, 1, requires_grad=True, device=device)
        t_bio = t * (t_max - t_min) + t_min

        C_bio = u_net(t_bio)
        L_pos = torch.relu(-C_bio).pow(2).mean()
        return L_pos 

    def set_all_seeds(seed: int, deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    # Makers for synthetic datasets with noise
    def make_replicates_noise(t, C_true, R=5, sigma0=0.2, alpha=1.0,
                          mode="power", seed=123, clip_min=0.0, dtype=np.float32):
        rng = np.random.default_rng(seed if seed is not None else None)
        t = np.asarray(t, dtype=dtype).reshape(-1)
        C_true = np.asarray(C_true, dtype=dtype).reshape(-1)
        T = C_true.shape[0]

        # prepare sigma0 per replicate
        if np.isscalar(sigma0):
            sigmas = float(sigma0) * np.ones(R, dtype=dtype)
        else:
            sigmas = np.asarray(sigma0, dtype=dtype)
            assert sigmas.shape[0] == R, "sigma0 must be scalar or length-R array"

        C_list = []
        for r in range(R):
            eps = rng.normal(loc=0.0, scale=1.0, size=T).astype(dtype) 
            s0 = float(sigmas[r])
            if mode == "add":
                noise = s0 * eps
                C_obs = C_true + noise

            elif mode == "mult":
                C_obs = C_true * (1.0 + s0 * eps)

            elif mode == "power":
                scale = s0 * (np.abs(C_true) ** float(alpha))
                noise = scale * eps
                C_obs = C_true + noise

            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'additive', 'multiplicative', or 'power'.")

            if clip_min is not None:
                C_obs = np.clip(C_obs, clip_min, None)

            C_list.append(C_obs.astype(dtype))

        return {"t": t, "C_list": C_list, "C_true": C_true}

    def split_train_val_test(t, C, train_ratio=0.6, val_ratio=0.2, seed=None):
        """
        Time-indexed split (keeps original chronological order).
        """
        assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
        n = len(t)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio  * n)
        idx_train = slice(0, n_train)
        idx_val   = slice(n_train, n_train + n_val)
        idx_test  = slice(n_train + n_val, n)
        return (t[idx_train], C[idx_train]), (t[idx_val], C[idx_val]), (t[idx_test], C[idx_test])

    def ensure_dir(d):
        os.makedirs(d, exist_ok=True)
        
    def gaussian_nll(y, mu, var):
        # Stable Gaussian NLL (up to +const)
        return 0.5 * ( (y - mu)**2 / var + torch.log(2 * torch.pi * var) ) ## added 2 pi on 26/3/26 

    def data_loss(u_net, noise_model, t_b, y_b, use_nll=True, mode=None):
        if use_nll:
            L, diag = BINNs.data_nll(u_net, noise_model, t_b, y_b, mode=mode)
            return L, diag
        else:
            mu = u_net(t_b)
            L = ((mu - y_b)**2).mean()
            return L, {"raw": (y_b - mu).detach(),
                    "std": (y_b - mu).detach(),  # placeholder not used
                    "sigma_eff": torch.zeros_like(mu),
                    "mu": mu.detach()}
        
    def data_nll(u_net, noise_model, t_b, y_b, mode):
        mu = u_net(t_b)
        sigma_eff, var = noise_model.sigma_and_var(mu)
        nll = BINNs.gaussian_nll(y_b, mu, var).mean()
        raw_res = (y_b - mu).detach()
        std_res = raw_res / torch.sqrt(var + 1e-12)
        return nll, {"raw": raw_res, "std": std_res, "sigma_eff": sigma_eff, "mu": mu.detach()}

    
    def trainBINN_ODE_g_multiRep_noise(
        Numrep,
        device,
        dataset,                
        save_dir,
        t_min, t_max,
        n_epochs, N_PDE, batch_size, learning_rate,
        u_num_layers, u_hidden_units, u_activation_fn,
        G_num_layers, G_hidden_units, G_activation_fn,
        u_weight, ode_weight, ic_weight, bio_weight,
        train_ratio=0.6, val_ratio=0.2,
        save_best_by="val_total",
        g_true_fn=None,
        band_mode="sd",
        seed=0, 
        noise_mode = "power",   
        resid_max_save = 5000, 
        use_nll = True, 
        data_sort = "random" 
    ):
        BINNs.set_all_seeds(seed)
        BINNs.ensure_dir(save_dir)
        rng = np.random.default_rng(seed if seed is not None else None)
        t_all = np.asarray(dataset["t"], dtype=np.float32).reshape(-1, 1)
        if "C_list" in dataset:
            C_list = [np.asarray(C, dtype=np.float32).reshape(-1, 1) for C in dataset["C_list"]]
        else:
            C_list = [np.asarray(dataset["C"], dtype=np.float32).reshape(-1, 1)]

        C_reps_np = np.stack([C.squeeze(-1) for C in C_list], axis=0)  
        R = len(C_list)
        n_total = len(t_all)
        splits = []
        sigma_runs = []         
        beta_runs = []          
        sigma_base_runs = []

        if data_sort == "time":
            fit_end = int(train_ratio * n_total)
            idx_fit = np.arange(fit_end)
            integ = max(1, int(1.0 / val_ratio))  
            idx_va = idx_fit[::integ]
            idx_tr = np.setdiff1d(idx_fit, idx_va)
            idx_te = np.arange(fit_end, n_total)

            for r in range(R):
                splits.append((idx_tr, idx_va, idx_te))
        else: 
            n_train = int(train_ratio * n_total)
            n_val   = int(val_ratio  * n_total)
            for r in range(R):
                perm = rng.permutation(n_total)
                idx_tr = perm[:n_train]
                idx_va = perm[n_train:n_train+n_val]
                idx_te = perm[n_train+n_val:]
                splits.append((idx_tr, idx_va, idx_te))
        Cmin = float(min(np.min(C) for C in C_list))
        Cmax = float(max(np.max(C) for C in C_list))
        C_grid = np.linspace(Cmin, Cmax, 400, dtype=np.float32).reshape(-1,1)
        C_grid_t = torch.as_tensor(C_grid, dtype=torch.float32, device=device)
        t_all_t_clean = torch.from_numpy(t_all).float().to(device)
        t0_val = float(np.min([np.min(t_all[splits[r][0]]) for r in range(R)]))
        C0_vals = []
        for r in range(R):
            C_r = C_list[r].squeeze()
            idx_t0 = int(np.argmin(np.abs(t_all.squeeze() - t0_val)))
            C0_vals.append(float(C_r[idx_t0]))
        C0_val = float(np.mean(C0_vals))
        C_full_runs, g_runs, evals = [], [], []
        val_runs = [] 
        resid_val_raw_by_rep = [[] for _ in range(R)]
        resid_val_std_by_rep = [[] for _ in range(R)]
        for rep in range(1, Numrep+1):
            rep_seed = seed + rep  
            BINNs.set_all_seeds(rep_seed)
            train_gen = torch.Generator(device="cpu")
            train_gen.manual_seed(rep_seed)
            save_dir_rep = os.path.join(save_dir, f"rep_{rep:02d}")
            BINNs.ensure_dir(save_dir_rep)
            t_tr_all, C_tr_all = [], []
            t_va_by_rep, C_va_by_rep = [], []
            for r in range(R):
                idx_tr, idx_va, _ = splits[r]
                t_tr_all.append(t_all[idx_tr])
                C_tr_all.append(C_list[r][idx_tr])
                t_va_by_rep.append(t_all[idx_va])
                C_va_by_rep.append(C_list[r][idx_va])
            t_tr_all = np.vstack(t_tr_all)
            C_tr_all = np.vstack(C_tr_all)
            t_va_all = np.vstack(t_va_by_rep)
            C_va_all = np.vstack(C_va_by_rep)
            if rep == 1:
                for r in range(R):
                    val_runs.append((
                        t_va_by_rep[r].reshape(-1).astype(np.float32),
                        C_va_by_rep[r].reshape(-1).astype(np.float32)
                    ))
            t_tr_t = torch.from_numpy(t_tr_all).float().to(device)
            C_tr_t = torch.from_numpy(C_tr_all).float().to(device)
            train_ds = TensorDataset(t_tr_t, C_tr_t)
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                generator=train_gen
            )
            t_va_t = torch.from_numpy(t_va_all).float().to(device)
            C_va_t = torch.from_numpy(C_va_all).float().to(device)
            u_net = UNet(u_num_layers, u_hidden_units, u_activation_fn, input_dim=1, output_dim=1).to(device)
            g_net = GNet(G_num_layers, G_hidden_units, G_activation_fn, input_dim=1, output_dim=1).to(device)
            noise_model = NoiseModel(mode=noise_mode).to(device)
            t0 = torch.tensor([[t0_val]], dtype=torch.float32, device=device, requires_grad=True)
            C0 = torch.tensor([[C0_val]], dtype=torch.float32, device=device)
            if use_nll:
                opt = torch.optim.Adam(
                    list(u_net.parameters()) + list(g_net.parameters()) + list(noise_model.parameters()),
                    lr=learning_rate
                )
            else:
                opt = torch.optim.Adam(
                    list(u_net.parameters()) + list(g_net.parameters()),
                    lr=learning_rate
                )
            best = {"score": float("inf")}
            ckpt_best = os.path.join(save_dir_rep, "binn_ode_g_best.pt")
            for epoch in range(1, n_epochs + 1):
                u_net.train(); g_net.train()
                for t_b, C_b in train_dl:
                    opt.zero_grad(set_to_none=True)
                    L_data, _ = BINNs.data_loss(u_net, noise_model, t_b, C_b, use_nll=use_nll, mode=noise_mode)
                    L_ode  = BINNs.ode_loss_multiplicative(t_min, t_max, device, u_net, g_net, N_PDE)
                    L_ic   = BINNs.initial_condition_loss(u_net, t0, C0)
                    L_bio = BINNs.bio_loss(u_net, t_min, t_max)
                    L = (u_weight  * L_data
                        + ode_weight * L_ode
                        + ic_weight  * L_ic
                        + bio_weight * L_bio)
                    L.backward()
                    opt.step()
                u_net.eval(); g_net.eval(); noise_model.eval()
                with torch.no_grad():
                    L_val_data, _ = BINNs.data_loss(u_net, noise_model, t_va_t, C_va_t, use_nll=use_nll, mode=noise_mode)
                    L_val_data = float(L_val_data.item())

                t_val_phys = t_va_t.clone().detach().requires_grad_(True)
                C_val_phys = u_net(t_val_phys)
                dCdt_val   = fcts.compute_dC_dt(C_val_phys, t_val_phys)
                L_val_ode  = torch.mean((dCdt_val - C_val_phys * g_net(C_val_phys))**2).item()

                L_val_total = u_weight*L_val_data + ode_weight*L_val_ode
                score = L_val_total if save_best_by == "val_total" else L_val_data
                if score < best["score"]:
                    best["score"] = score
                    torch.save({"u_net": u_net.state_dict(), "g_net": g_net.state_dict()}, ckpt_best)
                    torch.save(noise_model.state_dict(), os.path.join(save_dir_rep, "noise_best.pt"))

            if os.path.exists(ckpt_best):
                ckpt = torch.load(ckpt_best, map_location=device)
                u_net.load_state_dict(ckpt["u_net"]); g_net.load_state_dict(ckpt["g_net"])
                p_noise = os.path.join(save_dir_rep, "noise_best.pt")
                if os.path.exists(p_noise):
                    noise_model.load_state_dict(torch.load(p_noise, map_location=device))

            u_net.eval(); g_net.eval(); noise_model.eval()
            with torch.no_grad():
                C_hat_t   = u_net(t_all_t_clean)                            
                C_hat_full = C_hat_t.cpu().numpy().squeeze()
                g_hat      = g_net(C_grid_t).cpu().numpy().squeeze()

                _, var_full = noise_model.sigma_and_var(C_hat_t)  
                sigma_eff_full = torch.sqrt(var_full).cpu().numpy().squeeze()  

            for r in range(R):
                idx_tr, idx_va, _ = splits[r]
                t_va_r = torch.from_numpy(t_all[idx_va]).float().to(device)
                y_va_r = torch.from_numpy(C_list[r][idx_va]).float().to(device)
                _, diag = BINNs.data_nll(u_net, noise_model, t_va_r, y_va_r, mode=noise_mode)
                for key, store in (("raw", resid_val_raw_by_rep[r]), ("std", resid_val_std_by_rep[r])):
                    vals = diag[key].detach().cpu().numpy().reshape(-1)
                    if len(store) < resid_max_save:
                        take = int(min(resid_max_save - len(store), len(vals)))
                        store.extend(vals[:take])


            C_full_runs.append(C_hat_full)
            g_runs.append(g_hat)
            sigma_runs.append(sigma_eff_full)

            if getattr(noise_model, "beta_param", None) is not None:
                beta_runs.append(float(noise_model.beta_param.detach().cpu()))
            if getattr(noise_model, "log_sigma", None) is not None:
                sigma_base_runs.append(float(torch.exp(noise_model.log_sigma).detach().cpu()))

        C_full_runs = np.stack(C_full_runs, axis=0)  
        g_runs      = np.stack(g_runs,      axis=0)  
        sigma_full_runs = np.stack(sigma_runs, axis=0)  

        def _mean_band(arr, mode="sd", axis=0):
            mu = arr.mean(axis=axis)
            if mode == "sd":
                sd = arr.std(axis=axis)
                return mu, mu - sd, mu + sd
            elif mode == "iqr":
                q25 = np.quantile(arr, 0.25, axis=axis)
                q75 = np.quantile(arr, 0.75, axis=axis)
                return mu, q25, q75
            else:
                return mu, mu, mu

        C_full_mean, C_full_low, C_full_up = _mean_band(C_full_runs, mode=band_mode, axis=0)
        g_mean, g_low, g_up = _mean_band(g_runs, mode=band_mode, axis=0)
        sigma_full_mean, sigma_full_low, sigma_full_up = _mean_band(sigma_full_runs, mode=band_mode, axis=0)

        ensemble = {
            "Numrep": Numrep,
            "R": R,  
            "t_full": t_all.squeeze(),
            "C_full_runs": C_full_runs,             
            "C_full_mean": C_full_mean,              
            "C_full_low":  C_full_low,               
            "C_full_up":   C_full_up,                
            "C_grid": C_grid.squeeze(),              
            "g_runs": g_runs,                        
            "g_mean": g_mean, "g_low": g_low, "g_up": g_up,
            "val_runs": val_runs,                    
            "evals": evals,
            "resid_val_raw_by_rep": [np.asarray(x, dtype=np.float32) for x in resid_val_raw_by_rep],
            "resid_val_std_by_rep": [np.asarray(x, dtype=np.float32) for x in resid_val_std_by_rep],
            "noise_mode": noise_mode,
            "C_reps": C_reps_np,                          
            "sigma_full_runs": sigma_full_runs,          
            "sigma_full_mean": sigma_full_mean,          
            "sigma_full_low":  sigma_full_low,           
            "sigma_full_up":   sigma_full_up,             
            "beta_runs": np.asarray(beta_runs, dtype=np.float32) if len(beta_runs) else None,
            "sigma_base_runs": np.asarray(sigma_base_runs, dtype=np.float32) if len(sigma_base_runs) else None,
        }
        return ensemble
    
    def gather_sigma_beta(save_dir, Numrep=10, device="cpu"):
        sigmas = np.full((Numrep,), np.nan, dtype=np.float32)
        betas  = np.full((Numrep,), np.nan, dtype=np.float32)
        for rep in range(1, Numrep+1):
            p = os.path.join(save_dir, f"rep_{rep:02d}", "noise_best.pt")
            if not os.path.exists(p):
                continue
            try:
                nm = NoiseModel("power").to(device)
                nm.load_state_dict(torch.load(p, map_location=device))
                with torch.no_grad():
                    if getattr(nm, "log_sigma", None) is not None:
                        sigmas[rep-1] = float(torch.exp(nm.log_sigma).cpu().numpy())
                    if getattr(nm, "beta_param", None) is not None:
                        betas[rep-1]  = float(nm.beta_param.cpu().numpy())
            except Exception as e:
                print("Warning loading", p, ":", e)
        return sigmas, betas
    
    def run_on_dataset_and_save(npz_path, save_root, train_kwargs_override=None):
        data = np.load(npz_path, allow_pickle=True)
        t = data["t"]
        C_reps = data["C_reps"]   
        R = C_reps.shape[0]
        dataset = {"t": t, "C_list": [C_reps[r].astype(np.float32).reshape(-1) for r in range(R)]}
        base = Path(save_root)
        base.mkdir(parents=True, exist_ok=True)
        base_kwargs = dict(
            Numrep=10, device="cpu", dataset=dataset, save_dir=str(base),
            t_min=float(t.min()), t_max=float(t.max()),
            n_epochs=2000, N_PDE=256, batch_size=64, learning_rate=1e-3,
            u_num_layers=3, u_hidden_units=64, u_activation_fn=torch.nn.Tanh(),
            G_num_layers=3, G_hidden_units=64, G_activation_fn=torch.nn.Tanh(),
            u_weight=1.0, ode_weight=1.0, ic_weight=0, bio_weight=1.0,
            train_ratio=0.6, val_ratio=0.2, band_mode="sd", seed=123,
            noise_mode="power"
        )
        if train_kwargs_override:
            base_kwargs.update(train_kwargs_override)

        ensemble = BINNs.trainBINN_ODE_g_multiRep_noise(**base_kwargs)
        np.savez_compressed(os.path.join(base, "ensemble_summary.npz"),
                            C_full_mean = ensemble["C_full_mean"],
                            C_full_runs = ensemble["C_full_runs"],
                            g_mean = ensemble["g_mean"],
                            g_runs = ensemble["g_runs"],
                            sigma_full_mean = ensemble.get("sigma_full_mean"),
                            sigma_full_runs = ensemble.get("sigma_full_runs"),
                            beta_runs = ensemble.get("beta_runs"),
                            sigma_base_runs = ensemble.get("sigma_base_runs"),
                            meta = json.dumps({"npz_source": str(npz_path)})
                        )
        torch.save(ensemble, os.path.join(base, "ensemble_full.pt"))
        print(f"Training finished and ensemble saved to {base}/ensemble_summary.npz")

        return ensemble 
    
    def recompute_calibration_from_artifacts(save_dir, dataset_npz, Numrep=10, device="cpu", verbose=True):
        
        if not os.path.exists(dataset_npz):
            raise FileNotFoundError(f"Dataset file not found: {dataset_npz}")
        d = np.load(dataset_npz, allow_pickle=True)
        if "C_reps" not in d:
            raise KeyError("Dataset .npz must include 'C_reps' (shape R x T)")
        C_reps = d["C_reps"].astype(np.float32)   # (R, T)
        t = d["t"].astype(np.float32).ravel()
        R, T = C_reps.shape

        ens_path = os.path.join(save_dir, "ensemble_summary.npz")
        if not os.path.exists(ens_path):
            raise FileNotFoundError(f"ensemble_summary.npz not found at {ens_path}. "
                                    "Trainer must have saved C_full_runs for this function to work.")

        ens = np.load(ens_path, allow_pickle=True)
        if "C_full_runs" not in ens:
            raise KeyError("ensemble_summary.npz must contain 'C_full_runs' (Numrep x T).")
        C_full_runs = ens["C_full_runs"].astype(np.float32)  # (Numrep, T)
        if C_full_runs.shape[1] != T:
            raise ValueError(f"Time-length mismatch: dataset T={T}, C_full_runs shape {C_full_runs.shape}")

        fr1 = np.full((Numrep,), np.nan, dtype=np.float32)
        fr2 = np.full((Numrep,), np.nan, dtype=np.float32)
        raws = [None] * Numrep
        sigs = [None] * Numrep
        u_hats = [None] * Numrep

        for rep in range(1, Numrep+1):
            rep_idx = rep - 1
            if rep_idx >= C_full_runs.shape[0]:
                if verbose:
                    print(f"WARNING: C_full_runs contains only {C_full_runs.shape[0]} reps, skipping rep {rep}")
                continue

            u_hat = C_full_runs[rep_idx].ravel()   # (T,)
            u_hats[rep_idx] = u_hat

            if rep_idx >= R:
                if verbose:
                    print(f"WARNING: dataset has only {R} observed replicates; no observed replicate for rep {rep}. Skipping calibration for this rep.")
                continue

            y_obs = C_reps[rep_idx].ravel() 

            raw = (y_obs - u_hat).astype(np.float32)
            raws[rep_idx] = raw

            p_noise = os.path.join(save_dir, f"rep_{rep:02d}", "noise_best.pt")
            if not os.path.exists(p_noise):
                if verbose:
                    print(f"WARNING: noise checkpoint not found for rep {rep}: {p_noise}. Skipping.")
                continue

            try:
                nm = NoiseModel(mode="power").to(device)
                nm.load_state_dict(torch.load(p_noise, map_location=device))
                nm.eval()
                with torch.no_grad():
                    xt = torch.as_tensor(u_hat.reshape(-1,1), dtype=torch.float32, device=device)
                    sigma_eff_tensor, var_t = nm.sigma_and_var(xt)   
                    if var_t is None:
                        sigma_eff = sigma_eff_tensor.cpu().numpy().reshape(-1)
                    else:
                        sigma_eff = torch.sqrt(var_t).cpu().numpy().reshape(-1)
            except Exception as e:
                print(f"WARNING: failed to evaluate noise model for rep {rep}: {e}")
                continue

            sigs[rep_idx] = sigma_eff

            valid_mask = ~np.isnan(raw) & ~np.isnan(sigma_eff)
            if valid_mask.sum() == 0:
                fr1[rep_idx] = np.nan
                fr2[rep_idx] = np.nan
            else:
                fr1[rep_idx] = float(np.mean(np.abs(raw[valid_mask]) <= (sigma_eff[valid_mask] + 1e-12)))
                fr2[rep_idx] = float(np.mean(np.abs(raw[valid_mask]) <= (2.0*sigma_eff[valid_mask] + 1e-12)))

            if verbose:
                print(f"rep {rep:02d}: frac±1σ = {fr1[rep_idx]:.3f}, ±2σ = {fr2[rep_idx]:.3f}")

        out = {
            "1sigma": fr1, "2sigma": fr2,
            "raws": raws, "sigma_eff": sigs, "u_hats": u_hats,
            "t": t
        }
        return out

class NoiseModel(nn.Module):
    """
    mode:
    - 'add'        : Var = (sigma)^2
    - 'mult'  : Var = (sigma * |u|)^2
    - 'hetero_add'      : Var = (softplus(h(u)))^2
    - 'power'           : Var = (sigma * |u|^beta)^2    <-- learn beta
    """
    def __init__(self, mode="add", eps_min=1e-6, hidden=16):
        super().__init__()
        self.mode = mode
        self.eps_min = eps_min

        if mode in ("add", "mult"):
            self.log_sigma = nn.Parameter(torch.tensor(-1.0))
            self.beta_param = None
            self.head = None

        elif mode == "hetero_add":
            self.log_sigma = None
            self.beta_param = None
            self.head = nn.Sequential(
                nn.Linear(1, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1)
            )

        elif mode == "power":
            self.log_sigma = nn.Parameter(torch.tensor(-1.0))
            self.beta_param = nn.Parameter(torch.tensor(0.5)) 
            self.head = None

        else:
            raise ValueError("Unknown mode")

    def sigma_and_var(self, mu):
        mu = mu.detach()  
        if self.mode == "add":
            sigma = torch.exp(self.log_sigma) + self.eps_min
            var = sigma**2
            return sigma.expand_as(mu), var.expand_as(mu)

        if self.mode == "mult":
            sigma = torch.exp(self.log_sigma) + self.eps_min
            scale = torch.clamp(mu.abs(), min=self.eps_min)
            var = (sigma * scale)**2
            return sigma.expand_as(mu), var

        if self.mode == "hetero_add":
            s = F.softplus(self.head(mu)) + self.eps_min
            return s, s**2

        if self.mode == "power":
            sigma = torch.exp(self.log_sigma) + self.eps_min
            beta = self.beta_param
            scale = torch.clamp(mu.abs(), min=self.eps_min).pow(beta)
            var = (sigma * scale)**2
            return sigma.expand_as(mu), var

        raise RuntimeError("unreachable")
    
class SoftplusReLU(nn.Module):
    def __init__(self, threshold=20.0):
        super().__init__()
        self.threshold = threshold
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.where(x < self.threshold, self.softplus(x), self.relu(x))

class UNet(nn.Module):
    def __init__(self, num_layers, hidden_units, activation_fn, input_dim=1, output_dim=1):
        super(UNet, self).__init__()
        self.num_layers   = num_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn

        if hidden_units * num_layers == 1.0:
            self.hidden_layer = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(activation_fn)
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(activation_fn)
            layers.append(nn.Linear(hidden_units, output_dim))
            # Keep C positive (population), like your task-specific output heads
            layers.append(SoftplusReLU())
            self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        if hasattr(self, "network"):
            for m in self.network:
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
        else:
            init.xavier_normal_(self.hidden_layer.weight)
            if self.hidden_layer.bias is not None:
                init.zeros_(self.hidden_layer.bias)

    def forward(self, x):
        if self.hidden_units * self.num_layers == 1.0:
            return self.activation_fn(self.hidden_layer(x))
        else:
            return self.network(x)

class GNet(nn.Module):
    def __init__(self, num_layers, hidden_units, activation_fn, input_dim=1, output_dim=1):
        super(GNet, self).__init__()
        self.num_layers   = num_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn

        if hidden_units * num_layers == 1.0:
            self.hidden_layer = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(activation_fn)
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(activation_fn)
            layers.append(nn.Linear(hidden_units, output_dim))
            self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        if hasattr(self, "network"):
            for m in self.network:
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
        else:
            init.xavier_normal_(self.hidden_layer.weight)
            if self.hidden_layer.bias is not None:
                init.zeros_(self.hidden_layer.bias)

    def forward(self, x):
        if self.hidden_units * self.num_layers == 1.0:
            return self.activation_fn(self.hidden_layer(x))
        else:
            return self.network(x)

class plotters: 
    def plot_3x3_with_noise_BINN(ens_list, names=None, title_suffix="", show_cv=False):
        if names is None: names = [f"Model {i+1}" for i in range(len(ens_list))]
        assert len(ens_list) == 3

        WIDTH = 500.484 / 72.27
        HEIGHT = 1.55 * 3 + 1.9

        fig, axes = plt.subplots(3, 3, figsize =(WIDTH, HEIGHT), constrained_layout=True, squeeze=True) 
        for j, (ens, name) in enumerate(zip(ens_list, names)):
            t   = np.asarray(ens.get("t_full", ens.get("t"))).reshape(-1)
            Cmu, Clo, Cup = ens["C_full_mean"], ens["C_full_low"], ens["C_full_up"]
            Cgrid          = ens["C_grid"]
            gmu, glo, gup  = ens["g_mean"], ens["g_low"], ens["g_up"]
            C_reps         = ens.get("C_reps")
            sigma_mu       = ens.get("sigma_full_mean")
            sigma_lo       = ens.get("sigma_full_low")
            sigma_up       = ens.get("sigma_full_up")
            truth          = ens.get("_true", None)
            noise_mode     = ens.get("noise_mode", "?")


            true_log = fcts.logistic_analytic(t, r=0.08, K=10.0, C0=0.1)
            true_gom = fcts.gompertz_analytic(t, r=0.05, K=10.0, C0=0.1)
            true_rich = fcts.richards_analytic(t, r=0.06, K=10.0, C0=0.1, beta=2.0)

            ax = axes[0, j]
            if C_reps is not None:
                ax.plot(t, C_reps.T, alpha=0.12, color="#888888", marker = 'x', linestyle = 'None')
            if truth is not None:
                ax.plot(truth["t"], truth["C"], label="Truth")
            else:
                if j == 0:
                    ax.plot(t, true_log, ls="--", label="Truth")
                if j == 1:
                    ax.plot(t, true_gom, ls="--", label="Truth")
                if j == 2:
                    ax.plot(t, true_rich, ls="--", label="Truth`")
            ax.fill_between(t, Clo, Cup, alpha=0.2, label="Model band")
            ax.plot(t, Cmu, label="Model mean")
            ax.set_title(f"{name}")
            ax.set_xlabel("t")
            if j == 0: 
                ax.legend(loc="best")
                ax.set_ylabel("u")

            ax = axes[1, j]
            if truth is not None and truth.get("g_true_fn", None) is not None:
                g_true = truth["g_true_fn"](Cgrid)
                ax.plot(Cgrid, g_true, ls="--", label="g true")
            else:
                if j == 0:
                    ax.plot(Cgrid, fcts.logistic_g(Cgrid, r=0.08, K=10.0), ls="--", label="True g")
                if j == 1:
                    ax.plot(Cgrid, fcts.gompertz_g(Cgrid, r=0.05, K=10.0), ls="--", label="Trueg")
                if j == 2:
                    ax.plot(Cgrid, fcts.richards_g(Cgrid, r=0.06, K=10.0, beta=2.0), ls="--", label="True g")
            ax.fill_between(Cgrid, glo, gup, alpha=0.2, label="g band")
            ax.plot(Cgrid, gmu, label="g mean")
            ax.set_xlabel("u")
            ax.set_xlim(-0.5, 10.5)
            if j == 0: 
                ax.set_ylabel("g(u)")
                ax.legend(loc="best")

            ax = axes[2, j]
            if C_reps is not None:
                emp_sd = C_reps.std(axis=0, ddof=1)
                if show_cv:
                    emp = emp_sd / np.maximum(C_reps.mean(axis=0), 1e-12)
                    ylab = "CV = σ/|C|"
                else:
                    emp = emp_sd
                    ylab = "σ"
                ax.plot(t, emp, marker="o", ms=3, lw=0, alpha=0.6, label="Empirical")
            else:
                ylab = "σ"

            if sigma_mu is not None:
                pred = sigma_mu
                ax.plot(t, pred, label="Model prediction")
            else:
                ax.text(0.02, 0.80, "No learned noise stored", transform=ax.transAxes, fontsize=9)

            if j == 0:
                sigma0_true, alpha_true = 0.1, 0.0
                C_true = true_log
            elif j == 1:
                sigma0_true, alpha_true = 0.05, 0.5
                C_true = true_gom
            else:
                sigma0_true, alpha_true = 0.2, 1
                C_true = true_rich

            sigma_true = sigma0_true * np.maximum(C_true, 0.0) ** alpha_true

            ax.plot(t, sigma_true, ls="--", label="True noise")


            sigma_base_runs = ens.get("sigma_base_runs")
            beta_runs = ens.get("beta_runs")

            text_lines = []

            if sigma_base_runs is not None:
                sigma_base_mean = np.nanmean(sigma_base_runs)
                text_lines.append(rf"$\sigma_0 = {sigma_base_mean:.2f}$")

            if beta_runs is not None:
                beta_mean = np.nanmean(beta_runs)
                text_lines.append(rf"$\alpha = {beta_mean:.2f}$")

            if text_lines:
                ax.text(
                    0.95, 0.05,
                    "\n".join(text_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="none"
                    )
                )

            tag = "CV" if show_cv else "σ(t)"
            ax.set_xlabel("t")
            if j == 0: 
                ax.set_ylabel(ylab)
                ax.set_ylim(-0.01, 0.26)
                ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
                ax.legend(loc="best")
            if j == 1:
                ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
                ax.set_ylim(-0.01, 0.26)
        return fig
    
    def save_fig(fig, pathbase, formats=("pdf","png"), bbox_inches="tight"):
        os.makedirs(os.path.dirname(pathbase) or ".", exist_ok=True)
        for fmt in formats:
            path = f"{pathbase}.{fmt}"
            fig.savefig(path, bbox_inches=bbox_inches, pad_inches=0.02)
        print(f"Saved: {', '.join([f'{pathbase}.{f}' for f in formats])}")

    def plot_example_data(model_name, model_fn, t, alphas, sigma0s, R=10, seed=0, out_dir="./paper_figs/preview_plots_3x5"):
        os.makedirs(out_dir, exist_ok=True)
        n_rows = len(alphas)
        n_cols = len(sigma0s)
        WIDTH = 500.484 / 72.27
        HEIGHT = 1.55 * n_rows + 0.9
        fig, axes = plt.subplots(n_rows, n_cols,
        figsize=(WIDTH, HEIGHT),
        squeeze=False,
        sharex=True,
        sharey=False,
        constrained_layout=True)
        
        for i, alpha in enumerate(alphas):
            for j, sigma0 in enumerate(sigma0s):
                ds = BINNs.make_replicates_noise(t, model_fn, R=R, sigma0=sigma0, alpha=alpha, seed=seed + i*100 + j, clip_min=0.0)
                C_true = ds["C_true"]
                C_reps = np.stack(ds["C_list"], axis=0)  # (R, T)
                emp_mean = C_reps.mean(axis=0)
                emp_std = C_reps.std(axis=0, ddof=1)
                sigma_theoretical = sigma0 * (np.abs(C_true) ** alpha)
                
                ax = axes[i, j]
                for r in range(C_reps.shape[0]):
                    ax.plot(t, C_reps[r], color="#bbbbbb", alpha=0.5, label='replicates' if r == 0 else None)
                ax.plot(t, emp_mean, color="#E69F00", label="rep mean")
                ax.plot(t, C_true, color="k",ls="--", label="truth")
                if i == n_rows - 1:
                    ax.set_xlabel("t")
                else:
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_ylabel("u")
                else:
                    ax.set_ylabel("")
                if i == 0:
                    ax.set_title(rf"$\sigma_0={sigma0:.3f}$")
                if j == 0:
                    ax.annotate(
                        rf"$\alpha={alpha:.2f}$",
                        xy=(-0.9, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="center",
                        rotation=90,
                        fontsize=10
                    )

        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01) 
        )
        outpath = os.path.join(out_dir, f"{model_name}_examples_3x5.png")
        fig.savefig(outpath, dpi=300)
        print("Saved preview to:", outpath)
        return fig, outpath
    
    def compare_many_experiments(experiments, Numrep=10, device="cpu", n_grid=400, eps=1e-8, outpath=None, title=""):
        def _get_abs_grid(ens):
            if ens is None: return None
            mu = None
            if isinstance(ens, dict):
                mu = np.asarray(ens["C_full_mean"]).ravel()
            if mu is None:
                return None
            arr = np.asarray(mu).ravel()
            return np.clip(np.abs(arr), eps, None)
        

        grids = [_get_abs_grid(ex.get("ens")) for ex in experiments]
        grids_valid = [g for g in grids if g is not None]
        if len(grids_valid) == 0:
            xg = np.logspace(-6, 3, n_grid)
        else:
            mn = min(np.min(g) for g in grids_valid)
            mx = max(np.max(g) for g in grids_valid)
            mn = max(mn, eps)
            xg = np.unique(np.logspace(np.log10(mn), np.log10(mx + 1e-12), n_grid))

        def _eval_curves(save_dir):
            curves = []
            for rep in range(1, Numrep+1):
                p = os.path.join(save_dir, f"rep_{rep:02d}", "noise_best.pt")
                if not os.path.exists(p):
                    curves.append(np.full_like(xg, np.nan))
                    continue
                try:
                    nm = NoiseModel("power").to(device)
                    nm.load_state_dict(torch.load(p, map_location=device))
                    with torch.no_grad():
                        xt = torch.as_tensor(xg.reshape(-1,1), dtype=torch.float32, device=device)
                        _, var_t = nm.sigma_and_var(xt)
                        var_np = var_t.cpu().numpy().reshape(-1)
                    curves.append(var_np)
                except Exception as e:
                    print("Warning evaluating rep", rep, "in", save_dir, ":", e)
                    curves.append(np.full_like(xg, np.nan))
            return np.stack(curves, axis=0)

        curves_list = []
        mean_list = []
        p10_list = []
        p90_list = []
        betas_list = []
        sigmas_list = []

        for ex in experiments:
            save_dir = ex["save_dir"]
            curves = _eval_curves(save_dir)
            curves_list.append(curves)
            mean_list.append(np.nanmean(curves, axis=0))
            p10_list.append(np.nanpercentile(curves, 10, axis=0))
            p90_list.append(np.nanpercentile(curves, 90, axis=0))
            s,b = BINNs.gather_sigma_beta(save_dir, Numrep, device)
            sigmas_list.append(s); betas_list.append(b)

        fig, (ax1, ax2) = plt.subplots(1,2)
        for i, ex in enumerate(experiments):
            label = ex.get("label", f"exp{i}")
            color = ex.get("color")
            curves = curves_list[i]
            for k in range(curves.shape[0]):
                ax1.plot(xg, curves[k], color=color, alpha=0.12)
            ax1.plot(xg, mean_list[i], color=color, label=label)
            ax1.fill_between(xg, p10_list[i], p90_list[i], color=color, alpha=0.14)
            sigma_t = ex.get("sigma_true", None)
            beta_t = ex.get("beta_true", None)
            if (sigma_t is not None) and (beta_t is not None):
                true_var = (sigma_t * (xg**beta_t))**2
                ax1.plot(xg, true_var, color=color, ls='--', alpha=0.9)

        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("|û(t)|"); ax1.set_ylabel("Variance")
        ax1.legend(loc='lower right', fontsize=9)

        pos = []
        data_for_box = []
        labels_for_box = []
        colors_for_box = []
        spacing = 2.0
        cur_x = 1.0
        # for i, ex in enumerate(experiments):
        #     b = betas_list[i]
        #     s = sigmas_list[i]
        #     if not np.all(np.isnan(b)):
        #         data_for_box.append(b[~np.isnan(b)])
        #         pos.append(cur_x)
        #         labels_for_box.append(r'$\alpha$ ' + f'({ex.get("label")})')
        #         colors_for_box.append(ex.get("color"))
        #     cur_x += 0.9
        #     if not np.all(np.isnan(s)):
        #         data_for_box.append(s[~np.isnan(s)])
        #         pos.append(cur_x)
        #         labels_for_box.append(r'σ₀ ' + f'({ex.get("label")})')
        #         colors_for_box.append(ex.get("color"))
        #     cur_x += spacing

        # if len(data_for_box):
        #     bplots = ax2.boxplot(data_for_box, positions=pos, patch_artist=True, widths=0.6, showmeans=True)
        #     for patch, col in zip(bplots['boxes'], colors_for_box):
        #         patch.set_facecolor('white'); patch.set_edgecolor(col); patch.set_linewidth(1.4)
        #     for d,p,c in zip(data_for_box, pos, colors_for_box):
        #         jitter = np.random.normal(0, 0.03, size=len(d))
        #         ax2.scatter(np.full(len(d), p) + jitter, d, color=c, edgecolor='k', s=26)
        #     for i, ex in enumerate(experiments):
        #         sig_t = ex.get("sigma_true")
        #         bet_t = ex.get("beta_true")
        #         if bet_t is not None:
        #             lab = r'$\alpha$ ' + f'({ex.get("label")})'
        #             if lab in labels_for_box:
        #                 idx = labels_for_box.index(lab)
        #                 ax2.axhline(bet_t, color=colors_for_box[idx], ls='--')
        #         if sig_t is not None:
        #             lab2 = r'σ₀ (' + ex.get("label") + ')'
        #             if lab2 in labels_for_box:
        #                 idx2 = labels_for_box.index(lab2)
        #                 ax2.axhline(sig_t, color=colors_for_box[idx2], ls=':')
        #     ax2.set_xticks(pos)
        #     ax2.set_xticklabels(labels_for_box, rotation=40)
        #     ax2.grid(axis='y', ls=':', alpha=0.3)
        # else:
        #     ax2.text(0.5, 0.5, "No learned params found", ha='center')

        # plt.tight_layout()
        # if outpath:
        #     fig.savefig(outpath, dpi=300, bbox_inches='tight')
        #     print("Saved:", outpath)
        # plt.show()
        # ------------------------------------------------------------
        # Right panel: parameter recovery
        # ------------------------------------------------------------
        pos = []
        data_for_box = []
        labels_for_box = []
        colors_for_box = []
        kinds_for_box = []

        spacing = 2.2
        within_exp_gap = 0.85
        cur_x = 1.0

        for i, ex in enumerate(experiments):
            label = ex.get("label", f"exp{i}")
            color = ex.get("color")

            b = np.asarray(betas_list[i], dtype=float)
            s = np.asarray(sigmas_list[i], dtype=float)

            b = b[~np.isnan(b)]
            s = s[~np.isnan(s)]

            # -------------------------
            # alpha
            # -------------------------
            if len(b):
                data_for_box.append(b)
                pos.append(cur_x)
                labels_for_box.append(r'$\alpha$' + f' ({label})')
                colors_for_box.append(color)
                kinds_for_box.append("alpha")

            cur_x += within_exp_gap

            # -------------------------
            # sigma_0
            # -------------------------
            if len(s):
                data_for_box.append(s)
                pos.append(cur_x)
                labels_for_box.append(r'$\sigma_0$' + f' ({label})')
                colors_for_box.append(color)
                kinds_for_box.append("sigma")

            cur_x += spacing

        if len(data_for_box):

            # --------------------------------------------------------
            # Boxplots
            # --------------------------------------------------------
            bplots = ax2.boxplot(
                data_for_box,
                positions=pos,
                patch_artist=True,
                widths=0.55,
                showmeans=False,
                showfliers=False,
                medianprops=dict(
                    color="black",
                    linewidth=1.5
                ),
                whiskerprops=dict(
                    color="black",
                    linewidth=1.2
                ),
                capprops=dict(
                    color="black",
                    linewidth=1.2
                )
            )

            for patch, col in zip(
                bplots["boxes"],
                colors_for_box
            ):
                patch.set_facecolor("white")
                patch.set_edgecolor(col)
                patch.set_linewidth(1.5)

            # --------------------------------------------------------
            # Individual replicates
            #
            # alpha  = circles
            # sigma0 = squares
            # --------------------------------------------------------
            rng = np.random.default_rng(12345)

            for d, p, c, kind in zip(
                data_for_box,
                pos,
                colors_for_box,
                kinds_for_box
            ):
                jitter = rng.normal(
                    0,
                    0.055,
                    size=len(d)
                )

                marker = "o" if kind == "alpha" else "s"

                ax2.scatter(
                    np.full(len(d), p) + jitter,
                    d,
                    marker=marker,
                    color=c,
                    edgecolor="black",
                    linewidth=0.7,
                    s=42,
                    alpha=0.9,
                    zorder=4
                )

            # --------------------------------------------------------
            # Means = triangles
            # --------------------------------------------------------
            for d, p, c in zip(
                data_for_box,
                pos,
                colors_for_box
            ):
                ax2.scatter(
                    p,
                    np.mean(d),
                    marker="^",
                    color=c,
                    edgecolor="black",
                    linewidth=0.8,
                    s=95,
                    zorder=5
                )

            # --------------------------------------------------------
            # True values
            #
            # IMPORTANT:
            # Only draw a short segment around the relevant
            # experiment instead of a full-width axhline().
            # --------------------------------------------------------
            idx = 0

            for i, ex in enumerate(experiments):

                color = ex.get("color")
                beta_t = ex.get("beta_true")
                sigma_t = ex.get("sigma_true")

                # alpha position
                if idx < len(pos):
                    p_alpha = pos[idx]
                    idx += 1

                    if beta_t is not None:
                        ax2.plot(
                            [p_alpha - 1, p_alpha + 1],
                            [beta_t, beta_t],
                            color=color,
                            linestyle="--",
                            linewidth=3.0,
                            solid_capstyle="round",
                            zorder=2
                        )

                # sigma position
                if idx < len(pos):
                    p_sigma = pos[idx]
                    idx += 1

                    if sigma_t is not None:
                        ax2.plot(
                            [p_sigma - 1, p_sigma + 1],
                            [sigma_t, sigma_t],
                            color=color,
                            linestyle=":",
                            linewidth=3.0,
                            solid_capstyle="round",
                            zorder=2
                        )

            # --------------------------------------------------------
            # Axes
            # --------------------------------------------------------
            ax2.set_xticks(pos)
            ax2.set_xticklabels(
                labels_for_box,
                rotation=40,
                ha="right"
            )

            ax2.grid(
                axis="y",
                linestyle=":",
                alpha=0.3
            )

            # --------------------------------------------------------
            # Legend for marker meanings
            # --------------------------------------------------------
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=7,
                    label=r"$\alpha$ replicate"
                ),
                Line2D(
                    [0], [0],
                    marker="s",
                    linestyle="none",
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=7,
                    label=r"$\sigma_0$ replicate"
                ),
                Line2D(
                    [0], [0],
                    marker="^",
                    linestyle="none",
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=8,
                    label="mean"
                ),
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="--",
                    linewidth=3,
                    label=r"true $\alpha$"
                ),
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle=":",
                    linewidth=3,
                    label=r"true $\sigma_0$"
                ),
            ]

            ax2.legend(
                handles=legend_handles,
                loc="upper left",
                fontsize=8
            )

        else:
            ax2.text(
                0.5,
                0.5,
                "No learned params found",
                ha="center",
                va="center",
                transform=ax2.transAxes
            )
        plt.tight_layout()
        if outpath:
            fig.savefig(outpath, dpi=300, bbox_inches='tight')
            print("Saved:", outpath)
        plt.show()
        return fig, (xg, curves_list, mean_list, betas_list, sigmas_list)
    
    def mean_mech_rmse_for_save_dir(save_dir, ens_truth, Numrep=10, device="cpu"):
        runs, rmses, mean_curve = fcts.compute_mechanistic_forward_rmse(
            save_dir, ens_truth, Numrep=Numrep, device=device,
            use_ckpt_name="binn_ode_g_best.pt", initial_condition="ensemble_mean"
        )
        if rmses is None:
            return np.nan, runs, rmses, mean_curve
        good = ~np.isnan(rmses)
        if good.sum() == 0:
            return np.nan, runs, rmses, mean_curve
        return float(np.nanmean(rmses[good])), runs, rmses, mean_curve


def fit_power_noise_from_residuals_logreg(
    residuals,
    mu,
    min_abs_resid=1e-8,
    min_mu=1e-8,
    max_mu=None,
):
    """
    Fit sigma(mu) = sigma0 * |mu|^alpha from residuals using

        log|e| = log(sigma0) + alpha log|mu| + log|Z|,
        Z ~ N(0,1).

    Because E[log|Z|] = -(gamma + log(2))/2, the OLS intercept
    requires a correction to recover log(sigma0).

    Parameters
    ----------
    residuals : array-like
        Residuals e = y - mu, flattened.
    mu : array-like
        Corresponding fitted mean values.
    min_abs_resid : float
        Residuals smaller than this are excluded from the log fit.
    min_mu : float
        Mean values smaller than this are excluded.
    max_mu : float or None
        Optional upper cutoff on mu.

    Returns
    -------
    sigma0_hat : float
    alpha_hat : float
    details : dict
    """

    residuals = np.asarray(residuals, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()

    mask = (
        np.isfinite(residuals)
        & np.isfinite(mu)
        & (np.abs(residuals) > min_abs_resid)
        & (np.abs(mu) > min_mu)
    )

    if max_mu is not None:
        mask &= (np.abs(mu) <= max_mu)

    e = residuals[mask]
    u = np.abs(mu[mask])

    if len(e) < 5:
        raise ValueError(
            f"Too few valid residuals for noise fit: {len(e)}"
        )

    log_abs_e = np.log(np.abs(e))
    log_u = np.log(u)

    # OLS:
    # log|e| = intercept + alpha * log|u|
    alpha_hat, intercept_naive = np.polyfit(
        log_u,
        log_abs_e,
        1
    )

    # For Z ~ N(0,1):
    # E[log|Z|] = -(EulerGamma + log(2))/2
    euler_gamma = 0.5772156649015329
    correction = 0.5 * (euler_gamma + np.log(2.0))

    # intercept_naive =
    #     log(sigma0) + E[log|Z|]
    #
    # therefore:
    # log(sigma0) = intercept_naive + correction
    log_sigma0_hat = intercept_naive + correction
    sigma0_hat = np.exp(log_sigma0_hat)

    # fitted standard deviation
    sigma_hat = sigma0_hat * u**alpha_hat

    # Useful diagnostics
    log_pred = intercept_naive + alpha_hat * log_u
    ss_res = np.sum((log_abs_e - log_pred)**2)
    ss_tot = np.sum((log_abs_e - np.mean(log_abs_e))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(sigma0_hat), float(alpha_hat), {
        "n": len(e),
        "mask": mask,
        "residuals": e,
        "mu": u,
        "log_abs_residual": log_abs_e,
        "log_mu": log_u,
        "intercept_naive": float(intercept_naive),
        "log_sigma0": float(log_sigma0_hat),
        "sigma_hat": sigma_hat,
        "r2_logscale": float(r2),
    }


def fit_two_stage_noise_from_ensemble(
    ensemble,
    min_abs_resid=1e-8,
    min_mu=1e-8,
    max_mu=None,
):
    """
    Apply the direct residual-based power-law noise fit to every
    RMSE-trained BINN ensemble member.

    Expected ensemble fields:
        C_full_runs : (N_ens, T)
        C_reps      : (R, T)

    Returns
    -------
    dict containing sigma0 and alpha for each ensemble member,
    plus fitted sigma curves.
    """

    C_full_runs = np.asarray(ensemble["C_full_runs"], dtype=float)
    C_reps = np.asarray(ensemble["C_reps"], dtype=float)

    n_ens, T = C_full_runs.shape
    R, T_obs = C_reps.shape

    if T_obs != T:
        raise ValueError(
            f"Time mismatch: C_full_runs has T={T}, "
            f"but C_reps has T={T_obs}"
        )

    sigma_runs = []
    alpha_runs = []
    fit_details = []

    for k in range(n_ens):

        # Same fitted u(t) is compared against every observed replicate.
        mu = C_full_runs[k, :]

        # Shape:
        # residuals -> (R, T)
        residual_matrix = C_reps - mu[None, :]

        residuals = residual_matrix.ravel()
        mu_rep = np.broadcast_to(mu[None, :], residual_matrix.shape).ravel()

        sigma0_hat, alpha_hat, details = (
            BINNs.fit_power_noise_from_residuals_logreg(
                residuals=residuals,
                mu=mu_rep,
                min_abs_resid=min_abs_resid,
                min_mu=min_mu,
                max_mu=max_mu,
            )
        )

        sigma_runs.append(sigma0_hat)
        alpha_runs.append(alpha_hat)
        fit_details.append(details)

    sigma_runs = np.asarray(sigma_runs, dtype=np.float32)
    alpha_runs = np.asarray(alpha_runs, dtype=np.float32)

    mu_mean = np.mean(C_full_runs, axis=0)

    sigma_full_runs = np.asarray([
        sigma0 * np.maximum(np.abs(mu_mean), min_mu)**alpha
        for sigma0, alpha in zip(sigma_runs, alpha_runs)
    ], dtype=np.float32)

    return {
        "sigma_base_runs": sigma_runs,
        "beta_runs": alpha_runs,
        "sigma_full_runs": sigma_full_runs,
        "sigma_full_mean": np.mean(sigma_full_runs, axis=0),
        "fit_details": fit_details,
    }