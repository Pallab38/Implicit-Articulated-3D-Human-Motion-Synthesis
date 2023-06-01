### Copied from https://github.com/Pallab38/cvgLab_anr/blob/main/cvgLab_anr/latent_codes.py


from os import makedirs
from os.path import isdir, isfile, join
from typing import List

import torch

# pid_dict = {0:"377",1: "386",2:"387"}
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
        subjects: List[int],
        current_epoch: int,
        device,
        latent_texture_size: int = 512,
    ):
        self.device = device
        # print("self.device: ", self.device)
        self.data = {}
        self.latent_texture_size = latent_texture_size
        self.latent_code_path = join(training_path, "latent_codes")
        if not isdir(self.latent_code_path):
            assert (
                current_epoch == -1
            ), f"Cannot resume with empty latent code: {current_epoch}, from {self.latent_code_path}"
            makedirs(self.latent_code_path)
        if current_epoch == -1:
            # print("~~~~~~~~~~~~~~~~~~~~~~~")
            # print("create new latent codes")
            # print("~~~~~~~~~~~~~~~~~~~~~~~")

            base_latent = torch.FloatTensor(
                latent_texture_size, latent_texture_size, latent_dim
            ).uniform_()
            fname_base = join(self.latent_code_path, "base.pth")
            torch.save(base_latent, fname_base)

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
            for sid in subjects:
                # sid = subject.subject_id
                # sid = pid_dict[sid]
                # ensure that the subject in fact HAS data available!
                fname = join(path, f"latent_{sid}.pth")
                fname_optim = join(path, f"latent_{sid}_optim.pth")
                assert isfile(fname), fname
                assert isfile(fname_optim), fname_optim
                latent = torch.load(fname).to(self.device)
                print(f"Latent Code is loaded from: {fname}")
                print(f"Latent shape: {latent.shape}")
                latent.requires_grad = True
                optim = torch.optim.Adam(
                    [latent], lr=0.005, amsgrad=True, weight_decay=0
                )
                checkpoint = torch.load(fname_optim)
                optim.load_state_dict(checkpoint["optim"])
                self.data[sid] = {"latent": latent, "optim": optim}

    def save_all_codes(self, epoch=-1):
        if epoch > -1:
            epoch_dir = join(self.latent_code_path, "ep%06d" % epoch)
            assert not isdir(epoch_dir), epoch_dir
            makedirs(epoch_dir)
        for sid in self.data.keys():
            data = self.data[sid]
            latent = data["latent"]
            optim = data["optim"]
            fname = join(self.latent_code_path, f"latent_{sid}.pth")
            fname_optim = join(self.latent_code_path, f"latent_{sid}_optim.pth")
            torch.save({"optim": optim.state_dict()}, fname_optim)
            torch.save(latent, fname)

            if epoch > -1:
                fname = join(epoch_dir, f"latent_{sid}.pth")
                fname_optim = join(epoch_dir, f"latent_{sid}_optim.pth")
                torch.save({"optim": optim.state_dict()}, fname_optim)
                torch.save(latent, fname)

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
    # latent_dim =16
    # # training_path = "../my_data"
    # subjects = ["377","386","387"]
    # current_epoch = 0 - 1 ## self.current_epoch - 1
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("device: ", device)
    # latent_codes = LatentCodes(latent_dim,training_path,subjects,current_epoch,device)
    # latent_batch,latent_optims =latent_codes.get_codes_for_training(["377"])
    # print(latent_batch.shape)
    # print(latent_optims)

    #########  GET      LATENT CODES      FOR   EVALUATION   #########
    latent_dim = 3
    epoch = 100
    subjects = ["313"]
    root = "/home/user/model_imp_carv/ch3_logs/pulsarBasedNew2"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # latent_codes = LatentCodes(latent_dim=latent_dim, training_path=root,
    #                             subjects=subjects, current_epoch=epoch,
    #                             device=device, latent_texture_size=1024)
    # lt_codes = latent_codes.get_codes_for_evaluation(subjects=subjects)
    #### Directly access instead of using this class  ####
    sub = "313"
    lt_path = join(root, "latent_codes", "ep%06d" % epoch, f"latent_{sub}.pth")
    lt_codes = torch.load(lt_path)
    print(f"lt_codes.shape: {lt_codes.shape}")
