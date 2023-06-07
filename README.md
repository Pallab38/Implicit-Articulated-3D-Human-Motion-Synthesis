# Implicit-Articulated-3D-Human-Motion-Synthesis
Master's Thesis
## Abstract 
In this thesis, we address the problem of synthesizing human motion from sparse multi-view RGB videos. This is a challenging task since reconstructing a human from a set of sparse views is quite difficult because of extreme self-occlusions and the high variance of joint locations of the human body. Moreover, the appearance information has to be extracted too. We approach these challenges by presenting a novel approach that incorporates neural radiance field (NeRF)[[NeRF]](#nerf) defined in the 3D space with the SMPL body model [[SMPL]](#smpl). We also utilize per-subject latent codes for incorporating human body shape and appearance information. This formulation allows us to articulate pose information for unseen poses as well. Experiments on ZJU-MoCap dataset~\cite{peng2021neural} show that our approach performs competitively with the state-of-the-art approaches in terms of novel view synthesis. Furthermore, we demonstrate the capability of our method in terms of synthesizing novel poses.





## REFERENCE
<a id="Nerf">[nerf]</a>
Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren (2021).
Nerf: Representing scenes as neural radiance fields for view synthesis.
Communications of the ACM.
<a id="SMPL">[smpl]</a>
Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J (2015).
SMPL: A skinned multi-person linear model.
ACM transactions on graphics (TOG).
