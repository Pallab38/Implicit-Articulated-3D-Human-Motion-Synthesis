import torch 
import jax
import jax.numpy as jnp
import numpy as np 
import time 

torch.manual_seed(1310)

def np_mapping():
    np.random.seed(32)
    rng = np.random.default_rng(seed=12345)
    mat_B = rng.standard_normal((64,3))
    print(f"mat_B.shape: {mat_B.shape}")
    x = np.random.randn(1, 1024, 1024, 3)
    print(f"x.shape: {x.shape}")
    x_proj = (2. * np.pi * x) @ mat_B.T
    print(f"x_proj.shape: {x_proj.shape}")
    gamma_x = np.concatenate((np.cos(x_proj), np.sin(x_proj)), axis=-1)
    print(f"gamma_x: {gamma_x.shape}")

def input_mapping_torch(x: torch.Tensor, device:torch.device,
                        scale: float=10.0)-> torch.Tensor: 
    torch.manual_seed(1310)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x).to(device)
    elif not x.is_cuda:
        x = x.to(device)
    x = x.type(torch.float32)
    B = torch.randn(size=(256,3),device=device) * scale
    x = 2 * np.pi * x.clone()
    x_proj = x @ B.T
    gamma_x = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1).to(device)
    return gamma_x
    


def input_mapping(x: torch.Tensor, scale : float = 10.)-> torch.Tensor:

    # x = np.random.randn(1, 1024, 1024, 3)

    # x = jnp.array(x.cpu())
    x = jnp.array(x)
    # print(f"x.shape: {x.shape}")

    rand_key = jax.random.PRNGKey(seed=12345) 
    mat_B = jax.random.normal(rand_key,(256,3)) * scale
    # print(f"mat_B.shape: {mat_B.shape}")

    x_proj = (2. * jnp.pi * x) @ mat_B.T
    # print(f"x_proj.shape: {x_proj.shape}")
    gamma_x = jnp.concatenate((jnp.cos(x_proj), jnp.sin(x_proj)), axis=-1)
    # print(f"gamma_x: \nShape: {gamma_x.shape}, \
    #             \nType:{type(gamma_x)}")
    # print(f"x.dtype: {x.dtype}")
    return torch.from_numpy(np.array(gamma_x))

if __name__ =="__main__":
    # start = time.time()
    # np_mapping()
    # print(f"Time taken: {round(time.time()- start, 2)} s ")
    # start = time.time()
    x = torch.randn((1,1024,1024,3))
    enc_x = input_mapping(x)
    # print(f"Time taken: {round(time.time()- start, 2)} s ")
    print(f"enc_x: \nShape:{enc_x.shape}, \ndtype:{enc_x.dtype}")
    print(f"enc_x: {torch.min(enc_x).item(), torch.max(enc_x).item(), torch.sum(enc_x).item()}")
    gamma_x = input_mapping_torch(x)
    print(f"From torch: {gamma_x.shape, torch.min(gamma_x).item(), torch.max(gamma_x).item(), torch.sum(gamma_x).item() }")