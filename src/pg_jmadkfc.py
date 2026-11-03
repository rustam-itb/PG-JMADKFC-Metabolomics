"""PG-JMADKFC 1.0 -- locked research candidate.

Original JMADKFC loss + RGP safeguards + NPS-inspired *soft* decoded-prototype
completion with damped, detached updates after 50 joint epochs.

This does not use the original NPS-WKFCM standalone fitting or feature weighting.
This is NOT CGPC (no entropy-derived AE/prototype mixing). Models with different
completion semantics must not have their results pooled or called identical.
"""
import numpy as np
import torch
from rgp_jmadkfc import RGPJMADKFC

class PGJMADKFC(RGPJMADKFC):
    def __init__(self, *, burnin_epochs=50, completion_eta=0.25,
                 completion_clip=5.0, **kwargs):
        if int(burnin_epochs) < 0: raise ValueError('Invalid burnin_epochs')
        if not (0 < completion_eta <= 1): raise ValueError('Invalid completion_eta')
        if completion_clip <= 0: raise ValueError('Invalid completion_clip')
        self.burnin_epochs = int(burnin_epochs)
        self.completion_eta = float(completion_eta)
        self.completion_clip = float(completion_clip)
        self.dynamic_updates_ = 0
        self.completion_log_ = []
        self.completion_ = None
        self.observed_ = None
        self.M_ = None
        super().__init__(mode='candidate', **kwargs)

    def _prepare_inputs(self, X_missing, mask=None):
        # Discard unobserved values even if caller inadvertently passes a filled array.
        X = np.asarray(X_missing, dtype=float)
        M = np.isfinite(X) if mask is None else np.asarray(mask, dtype=bool)
        if X.ndim != 2 or X.shape != M.shape or not np.isfinite(X[M]).all():
            raise ValueError('Invalid inputs/mask')
        Xsafe = np.where(M, X, np.nan)
        Xzero, Mfloat, Xt = super()._prepare_inputs(Xsafe, M)
        self.M_ = M.copy()
        self.observed_ = np.asarray(Xzero, dtype=np.float64).copy()
        self.completion_ = self.observed_.copy()
        return Xzero, Mfloat, Xt

    def _update_dynamic_completion(self, X_tilde_tensor, mask_tensor, epoch, U, V):
        if epoch + 1 <= self.burnin_epochs:
            return
        with torch.no_grad():
            self.model_.eval()
            Z_tensor, _ = self.model_(X_tilde_tensor)
            Z = Z_tensor.detach().cpu().numpy()
            sigma_i = self._sample_bandwidths_np(Z)
            sigma_k = self._prototype_bandwidths_np(U, sigma_i)
            D, _ = self._adaptive_kernel_np(Z, V, sigma_i, sigma_k)
            U_new = self._update_membership_np(D)
            # Soft-NPS inspired completion: decoded prototypes weighted by U^m.
            W = np.power(np.clip(U_new, 1e-12, 1), self.alpha)
            W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
            P = W @ self._decode_numpy(np.asarray(V, dtype=float))
            P = np.clip(P, -self.completion_clip, self.completion_clip)
            if not np.isfinite(P).all():
                raise FloatingPointError('Non-finite prototype candidate')
            M = self.M_
            old = self.completion_
            new = old.copy()
            new[~M] = ((1-self.completion_eta)*old + self.completion_eta*P)[~M]
            new[M] = self.observed_[M]
            if not np.array_equal(new[M], self.observed_[M]):
                raise AssertionError('Observed intensities were changed')
            if not np.isfinite(new).all():
                raise FloatingPointError('Non-finite completed input')
            self.completion_ = new
            # In-place update only the intensity features; keep the original mask input.
            X_tilde_tensor[:, :new.shape[1]].copy_(
                torch.as_tensor(new, device=X_tilde_tensor.device,
                                dtype=X_tilde_tensor.dtype))
            self.dynamic_updates_ += 1
            if epoch in (49, 50, 74, 99, 149, 198):
                self.completion_log_.append({
                    'epoch':int(epoch+1), 'eta':self.completion_eta,
                    'mean_abs_delta_missing':float(np.mean(np.abs(new[~M]-old[~M]))),
                    'membership_entropy':float(np.mean(-np.sum(U_new*np.log(np.clip(U_new,1e-12,1)),axis=1))/np.log(2))
                })

    def fit(self, X_missing, mask=None):
        super().fit(X_missing, mask)
        if not np.array_equal(self.completion_[self.M_], self.observed_[self.M_]):
            raise AssertionError('Observed data changed during fitting')
        if self.dynamic_updates_ != max(0, self.joint_epochs - 1 - self.burnin_epochs):
            raise AssertionError('Unexpected completion schedule')
        return self
