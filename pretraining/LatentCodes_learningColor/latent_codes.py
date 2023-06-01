### Copied from https://github.com/Pallab38/cvgLab_anr/blob/main/cvgLab_anr/latent_codes.py


from os import makedirs
from os.path import isdir, isfile, join
from typing import List

import torch

"""
Creates "latent_codes" for a given subject(e.g. 313) and initials the latentCodes (weights) to the ADAM Optimizer

"""


pid_dict = {
    0: "313",
    1: "315",
    2: "377",
    3: "386",
    4: "387",
    5: "390",
    6: "392",
    7: "393",
    8: "394",
}


class LatentCodes:
    def __init__(
        self,
        latent_dim: int,
        training_path: str,
        subjects: List[str],
        current_epoch: int,
        device: torch.device,
        latent_texture_size: int = 512,
    ):
        self.device = device
        # print("self.device: ", self.device)
        self.data = {}
        self.latent_texture_size = latent_texture_size
        self.latent_code_path = join(training_path, "latent_codes")
        self.latent_img_path = join(training_path, "latent_images")

        if not isdir(self.latent_code_path):
            assert (
                current_epoch == -1
            ), f"Cannot resume with empty latent code: {current_epoch}, from {self.latent_code_path}"
            makedirs(self.latent_code_path)
        if not isdir(self.latent_img_path):
            assert (
                current_epoch == -1
            ), f"Cannot resume with empty latent code: {current_epoch}, from {self.latent_img_path}"
            makedirs(self.latent_img_path)

        if current_epoch == -1:
            # print("~~~~~~~~~~~~~~~~~~~~~~~")
            # print("create new latent codes")
            # print("~~~~~~~~~~~~~~~~~~~~~~~")

            base_latent = torch.FloatTensor(
                latent_texture_size, latent_texture_size, latent_dim
            ).uniform_()
            fname_base = join(self.latent_code_path, "base.pth")
            torch.save(base_latent, fname_base)

            # base_latent_image =  torch.FloatTensor(300000, latent_dim).uniform_()
            # base_lt_path = join(self.latent_img_path, "base_image.pth")
            # torch.save(base_latent_image, base_lt_path)

            for sid in subjects:
                # sid = subject.subject_id
                # sid = pid_dict[sid]
                # loading the tensor from file to ENSURE that we do NOT share any gradients!
                latent = torch.load(fname_base).to(self.device)
                latent.requires_grad = True
                optim = torch.optim.Adam(
                    [latent], lr=0.05, amsgrad=True, weight_decay=0
                )

                self.data[sid] = {"latent": latent, "optim": optim}

        else:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print("load existing latent codes")
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            path = join(self.latent_code_path, "ep%06d" % current_epoch)
            # lt_img_path = join(self.latent_img_path, "ep%06d" % current_epoch)
            for sid in subjects:
                # sid = subject.subject_id
                # sid = pid_dict[sid]
                # ensure that the subject in fact HAS data available!
                fname = join(path, f"latent_{sid}.pth")
                fname_optim = join(path, f"latent_{sid}_optim.pth")
                # fname_image = join(lt_img_path, f"latent_{sid}.pth")
                assert isfile(fname), fname
                assert isfile(fname_optim), fname_optim
                # assert isfile(fname_image), fname_image
                latent = torch.load(fname).to(self.device)
                latent.requires_grad = True
                optim = torch.optim.Adam(
                    [latent], lr=0.005, amsgrad=True, weight_decay=0
                )
                checkpoint = torch.load(fname_optim)
                optim.load_state_dict(checkpoint["optim"])
                self.data[sid] = {"latent": latent, "optim": optim}

    def save_all_codes(self, latent_image, epoch=-1):
        """
        Save Latent Codes and Image after each epoch.\n
        Args:
            latent_image: Output of `torch.nn.functional.grid_sample(latent,pts_uv,align_corners=True)`
                         and the shape is [bs, num_pts, 3]
        """
        if epoch > -1:
            epoch_dir = join(self.latent_code_path, "ep%06d" % epoch)
            assert not isdir(epoch_dir), epoch_dir
            makedirs(epoch_dir)
            ### For latent image
            lt_img_dir = join(self.latent_img_path, "ep%06d" % epoch)
            assert not isdir(lt_img_dir), lt_img_dir
            makedirs(lt_img_dir)

        for sid in self.data.keys():
            data = self.data[sid]
            latent = data["latent"]
            optim = data["optim"]
            fname = join(self.latent_code_path, f"latent_{sid}.pth")
            fname_optim = join(self.latent_code_path, f"latent_{sid}_optim.pth")
            torch.save(
                {"optim": optim.state_dict()}, fname_optim
            )  ### optimizer dictionary
            torch.save(latent, fname)  ## latent weights

            fname_lt_img = join(self.latent_img_path, f"lt_img_{sid}.pth")
            # latent_image = data["latent_image"]
            torch.save(latent_image, fname_lt_img)
            # print(f"LatentCodes: latent_img sum: {torch.sum(latent_image)}")

            if epoch > -1:
                fname = join(epoch_dir, f"latent_{sid}.pth")
                fname_optim = join(epoch_dir, f"latent_{sid}_optim.pth")
                torch.save({"optim": optim.state_dict()}, fname_optim)
                torch.save(latent, fname)

                fname_lt_img = join(lt_img_dir, f"lt_img_{sid}.pth")
                torch.save(latent_image, fname_lt_img)
                # print(f"LatentCodes: latent_img sum: {torch.sum(latent_image)},\
                #         shape: {latent_image.shape}")

    def get_codes_for_evaluation(self, subjects: List[int]):
        Latent = []
        for sid in subjects:
            # sid = sid.item()
            # sid = pid_dict[sid]
            data = self.data[sid]
            latent = data["latent"]
            Latent.append(latent.unsqueeze(0))

        Latent = torch.cat(Latent, dim=0)
        return Latent

    def get_codes_for_training(self, subjects: List[int]):
        """
        return the latent vectors and the respective optims.
            Sids can be repeated - the optimizer will then be only return ones for this pair
        """
        Latent = []
        Optim = []
        already_used_sids = set()
        for sid in subjects:
            # sid = sid.item()
            data = self.data[sid]
            latent = data["latent"]
            optim = data["optim"]
            Latent.append(latent.unsqueeze(0))
            if sid not in already_used_sids:
                Optim.append(optim)
                already_used_sids.add(sid)

        Latent = torch.cat(Latent, dim=0)
        return Latent, Optim


if __name__ == "__main__":
    latent_dim = 16
    training_path = "../my_data"
    subjects = ["377", "386", "387"]
    current_epoch = 0 - 1  ## self.current_epoch - 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    latent_codes = LatentCodes(
        latent_dim, training_path, subjects, current_epoch, device
    )
    latent_batch, latent_optims = latent_codes.get_codes_for_training(["377"])
    print(latent_batch.shape)
    print(latent_optims)
