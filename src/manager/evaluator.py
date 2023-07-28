import tqdm
import torch
import torch.nn.functional as F


class Evaluator():
    def __init__(self, args):
        self.args = args

        # add some metrics to evaluate the model, e.g. MRR / f1 score
        self.metrics = []

    @torch.no_grad()
    def evaluate(self, model, dataloader, device):
        
        model.eval()

        eval_res = {metric: [] for metric in self.metrics}

        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, batch in enumerate(dataloader):
                batch_data, batch_label = batch
                
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                score = model(batch_data)

                batch_metric = self.compute_metric(score, batch_label)

                for metric in self.metrics:
                    eval_res[metric].append(batch_metric[metric])

                pbar.update(1)
        
        eval_res = {k: np.mean(v) for k,v in eval_res.items()}

        return eval_res
    
    def compute_metric(self, pred, ground_truth):
        pass
