import torch



class PytorchModel:
    def __init__(self, model_class, model_path, device):
        self.model_class = model_class
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model = self.model_class(device=self.device)
        if self.model_path is not None:
            ckpt = torch.load(self.model_path, map_location=torch.device('cpu'))
            if ckpt:
                model.load_state_dict(ckpt["model_state_dict"])

        model.eval()
        return model
