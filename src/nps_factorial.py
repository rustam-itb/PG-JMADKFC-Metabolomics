"""2x2 NPS component isolation on the exact prior RGP-JMADKFC architecture.

cell (completion, weighting): (0,0) plain RGP, (1,0) NPS input
completion, (0,1) NPS-estimated feature weights in observed-only MSE,
(1,1) original RGP+NPS. Features weights are computed by fitting the same
NPSWKFCMReimplementation model in both weighted cells. NPS fit estimates
prototypes on data with missing entries; in weighting-only cells its *completed
values are discarded*, but its weights necessarily depend on its fitting steps.
None of the fit paths uses external class labels.
"""
import numpy as np
import torch
from rgp_jmadkfc import RGPJMADKFC
from source_reimplementations import NPSWKFCMReimplementation

class FactorialNPSJMADKFC(RGPJMADKFC):
    def __init__(self,*,nps_completion=False,feature_weighting=False,**kwargs):
        self.nps_completion=bool(nps_completion)
        self.feature_weighting=bool(feature_weighting)
        self.nps_model_=None
        self.nps_feature_weight_=None
        super().__init__(mode='candidate',**kwargs)

    def _prepare_inputs(self,X_missing,mask=None):
        X=np.asarray(X_missing,dtype=float)
        M=np.isfinite(X) if mask is None else np.asarray(mask,dtype=bool)
        if X.shape!=M.shape or X.ndim!=2:
            raise ValueError('Shape mismatch')
        if self.nps_completion or self.feature_weighting:
            nps=NPSWKFCMReimplementation(n_clusters=self.n_clusters,
                        random_state=self.random_state,max_iter=80).fit(X,M)
            self.nps_model_=nps
            if self.feature_weighting:
                self.nps_feature_weight_=np.maximum(nps.feature_weights_.mean(axis=0),1e-10)
                self.nps_feature_weight_/=self.nps_feature_weight_.mean()
                self.nps_feature_weight_=np.clip(self.nps_feature_weight_,.1,10.)
            completed=nps.X_completed_ if self.nps_completion else np.where(M,X,0.)
        else:
            completed=np.where(M,X,0.)
        # Reuses same original RGP input centering, encoder and safeguards.
        return super()._prepare_inputs(completed,M)

    def _masked_reconstruction_loss(self,Xzero,M,Xhat):
        if not self.feature_weighting:
            return super()._masked_reconstruction_loss(Xzero,M,Xhat)
        w=torch.as_tensor(self.nps_feature_weight_,device=Xzero.device,dtype=Xzero.dtype)
        return (M*w*(Xzero-Xhat).square()).sum()/(M*w).sum().clamp_min(1e-10)
